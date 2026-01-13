# analysis utilities
import torch
import pandas as pd

from tqdm import tqdm

from datasets import DatasetDict
from torch.utils.data import IterableDataset

from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon


def run_across_layers(model, dataset, analysis, layer_template, max_ind,
                      **analysis_kwargs):
    """Runs an analysis for all layers in a model """
    results = []
    print(f'📕 Analysis: {analysis}')
    for layer_ind in range(max_ind):
        layer_name = layer_template.format(ind=layer_ind)
        print(f'📄 Layer: {layer_name}')

        stats = analysis(model, layer_name, dataset, **analysis_kwargs)
        results.append({'name': layer_name} | stats)

    return pd.DataFrame(results)


# Simultaneous randomization and readout
def probe_accuracy_post_randomization(model, dataset, layer_template, max_ind,
                                      write_out=True, write_backups=False, **probe_kwargs):
    """Add a randomization hook to model layers, then evaluate the accuacy of
    decoders on various layers of the model
    """
    all_results = []

    for layer_ind in range(max_ind):
        layer_name = layer_template.format(ind=layer_ind)
        print(f'Randomized: {layer_name}')
        handle = add_randomization_hook(model, layer_name, randomization_mode='shuffle')

        current_results = run_across_layers(model, dataset, linear_probe_by_ridge_regression,
                                            layer_template, max_ind, **probe_kwargs
                                            )

        current_results = current_results.rename(columns={'name': 'probed_layer'})
        current_results['randomized_layer'] = layer_name

        all_results.append(current_results)
        handle.remove()

        if write_backups:
            current_results.to_csv(f'intermediate_{layer_name}.csv')

    # This serves as a control on randomization hooks that haven't been removed
    baseline_results = run_across_layers(model, dataset, linear_probe_by_ridge_regression,
                                         layer_template, max_ind, **probe_kwargs
                                         )
    baseline_results = baseline_results.rename(columns={'name': 'probed_layer'})
    baseline_results['randomized_layer'] = 'none'

    all_results.append(baseline_results)

    all_results = pd.concat(all_results, ignore_index=True)

    if write_out:
        all_results.to_csv('randomization_scan.csv')

    return all_results


# Layer randomization 
def accuracy_random_CLS(model, layer_name, dataset, randomization_mode='shuffle'):
    """ Randomize the class tokens in the specified layer, compute accuracy
    """
    if layer_name is not None:
        handle = add_randomization_hook(model, layer_name,
                                        randomization_mode=randomization_mode)

    outputs = model_accuracy(model, dataset["test"])

    if layer_name is not None:
        handle.remove()

    return {'accuracy': outputs, 'test': 'cls_randomization'}


def add_randomization_hook(model, layer_name, randomization_mode='shuffle'):
    """Randomizes CLS tokens """

    module_dict = {k: v for k, v in model.named_modules()}
    module = module_dict[layer_name]

    def randomization_hook(module, input, output):
        copied_output = output.clone()
        batch, tokens, dims = copied_output.shape
        
        if randomization_mode == 'shuffle':
            ind_perm = torch.randperm(batch)
            copied_output[:, 0, :] = copied_output[ind_perm, 0, :]
        elif randomization_mode == 'normal':
            # default to normal
            copied_output[:, 0, :] = torch.randn(batch, dims)

        return copied_output

    handle = module.register_forward_hook(randomization_hook)
    return handle


# Layer decoding
def linear_probe_by_ridge_regression(base_model, layer_name, dataset,
                                     samples_per_label=200,
                                     **embeddingKwargs):
    """Liner probing to assess class identity presence"""

    embeddingKwargs = {'device': 'mps', 'batch_size': 64} | embeddingKwargs
    embedding_datasets, hook_manager = create_cls_embedding_datasets(
        base_model=base_model,
        dataset_dict=dataset,
        layer_name=layer_name,
        **embeddingKwargs
    )

    num_train_points = samples_per_label * base_model.num_labels
    num_batches = num_train_points / embeddingKwargs['batch_size']

    def unpack_dataset(dataset):
        images = []
        labels = []

        ind = 0
        for batch in tqdm(iter(dataset), total=min(num_batches, len(dataset))):
            if ind >= num_batches:
                break
            images.append(batch[0].cpu())
            labels.append(batch[1].cpu())

            ind += 1
        
        images = torch.concat(images)
        labels = torch.concat(labels)

        return images, labels

    train_embeddings, train_labels = unpack_dataset(embedding_datasets["train"])
    test_embeddings, test_labels = unpack_dataset(embedding_datasets["test"])

    encoder = OneHotEncoder()
    targets = 2*encoder.fit_transform(train_labels.reshape(-1, 1)).toarray() - 1

    regressor = Ridge()
    best_reg_model = RandomizedSearchCV(
                                estimator=regressor, 
                                param_distributions={'alpha': expon(scale=100)},
                                n_iter=15,
                                cv=5, 
                                scoring='neg_mean_squared_error',
                                refit=True
                              )

    best_reg_model.fit(train_embeddings, targets)

    return {'accuracy':
            (best_reg_model.predict(test_embeddings).argmax(1) == test_labels).float().mean().item(),
            'test': 'layerwise_probe'
            }


def apply_model_decoder(model, layer_name, dataset, **embeddingKwargs):
    """ Uses the model's classifier to decode CLS tokens in given layer
    Measures accuracy.
    """
    # device management
    if 'device' in embeddingKwargs:
        model.to(embeddingKwargs['device'])
    else:
        embeddingKwargs['device'] = model.device

    decoder = model.classifier

    embedding_datasets, hook_manager = create_cls_embedding_datasets(
        base_model=model,
        dataset_dict=dataset,
        layer_name=layer_name,
        **embeddingKwargs
    )

    with hook_manager:
        stats = classification_stats(decoder, embedding_datasets['test'],
                                     batch_size=None, shuffle=False)
        stats['test'] = 'layerwise_model_decoder'

        return stats


def create_cls_embedding_datasets(base_model, dataset_dict, layer_name,
                                  batch_size=64, seed=42, device=None, **kwargs):
    """Create train/valid/test CLSEmbeddingDataset instances sharing a single hook.
    Args:
        base_model: The base model to attach the hook to
        dataset_dict: Dictionary with keys 'train', 'valid', 'test' containing base datasets
        layer_name: Name of the layer to attach the hook to
        **kwargs: Additional arguments passed to CLSEmbeddingDataset
    
    Returns:
        Dictionary with keys 'train', 'valid', 'test' containing CLSEmbeddingDataset instances
    """
    # Create a single shared hook manager
    hook_manager = SharedHookManager(base_model, layer_name)
    
    # Create datasets sharing the same hook manager
    datasets = DatasetDict()
    for split in ['train', 'valid', 'test']:
        if split in dataset_dict:
            datasets[split] = CLSEmbeddingDataset(
                base_model=base_model,
                base_inputs=dataset_dict[split],
                layer_name=layer_name,
                batch_size=batch_size,
                shuffle=(True if split == 'train' else False),
                seed=seed,
                device=device,
                hook_manager=hook_manager,
                **kwargs
            )
    
    return datasets, hook_manager


class SharedHookManager:
    """Manages a single forward hook shared across multiple datasets"""
    def __init__(self, base_model, layer_name):
        self.CLS_tokens = None
        module_dict = {k: v for k, v in base_model.named_modules()}
        self.hook_handle = module_dict[layer_name].register_forward_hook(self.recording_hook)
    
    def recording_hook(self, module, input, output):
        self.CLS_tokens = output[:, 0, :].detach().clone().cpu()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup the hook"""
        self.cleanup()
        return False  # Don't suppress exceptions
    
    def cleanup(self):
        """Explicitly remove the hook. Should be called when done."""
        if hasattr(self, 'hook_handle') and self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
        self.CLS_tokens = None
    
    def __del__(self):
        """Cleanup the hook as fallback"""
        self.cleanup()


class CLSEmbeddingDataset(IterableDataset):
    """Dateset of CLS token embeddings for elements of a base dataset
    """
    def __init__(self, base_model, base_inputs,
                 layer_name='embedding.vit.layernorm',
                 hook_manager=None, device='cpu',
                 batch_size=128, shuffle=True, seed=42
                 ):
        super(CLSEmbeddingDataset).__init__()

        self.base_model = base_model.to(device)
        self.base_inputs = base_inputs

        # Create a new hook manager if none provided
        if hook_manager is None:
            self.hook_manager = SharedHookManager(base_model, layer_name)
        else:
            self.hook_manager = hook_manager

        # Batching parameters
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        self.gen = torch.Generator()
        self.gen.manual_seed(seed)
    
    @property
    def CLS_tokens(self):
        """Access CLS_tokens from the hook manager"""
        return self.hook_manager.CLS_tokens

    def __iter__(self):
        if self.shuffle:
            inds = torch.randperm(len(self.base_inputs), generator=self.gen)
        else:
            inds = torch.arange(len(self.base_inputs))

        batches = [inds[i: i+self.batch_size].tolist() 
                   for i in range(0, len(self.base_inputs), self.batch_size)
                   ]

        try:
            for batch_inds in batches:
                # huggingface dataset: batches are dictionaries
                batch = self.base_inputs[batch_inds]

                labels = torch.as_tensor(batch.pop('labels'))
                labels = labels.flatten()
                
                batch_on_device = {k: v.to(self.device) for k, v in batch.items()}

                _ = self.base_model.forward(**batch_on_device)
                activity = self.CLS_tokens

                yield activity, labels.cpu()

        finally:
            # Reset CLS_tokens to None at the end of iteration for safety
            self.hook_manager.CLS_tokens = None

    def __len__(self):
        return len(self.base_inputs) // self.batch_size + 1


# Accuracy metrics

def model_accuracy(model, dataset, device='mps', **evalkwargs):
    """ Accuracy evaluation for huggingface models
    """
    model = model.eval().to(device)

    defaults = {'batch_size': 128, 'shuffle': True}

    dl = torch.utils.data.DataLoader(dataset, **(defaults | evalkwargs))
    outputs = []

    with torch.no_grad():
        for batch in tqdm(iter(dl)):
            labels = batch.pop('labels')

            batch_on_device = {k: v.to(model.device) for k, v in batch.items()}
            preds = model.forward(**batch_on_device).logits.argmax(1).cpu()

            outputs.append(preds.cpu() == labels)

    return torch.concat(outputs).float().mean().item()


def classification_stats(model, dataset, device=None, **evalkwargs):
    """ Accuracy and identifiability evaluation for raw pytorch models"""
    model = model.eval().to(device)

    if device is None:
        # robustly extract model device
        device = next(model.parameters()).device

    defaults = {'batch_size': 128, 'shuffle': True}
    evalkwargs = (defaults | evalkwargs)

    dl = torch.utils.data.DataLoader(dataset, **evalkwargs)
    ranks = []

    if evalkwargs['batch_size'] is None:
        total = len(dl)
    else:
        total = len(dl) / evalkwargs['batch_size'] 

    with torch.no_grad():
        for batch in tqdm(iter(dl), total=total):
            images, labels = batch
            logits = model.forward(images.to(device)).cpu()
            labels = labels.cpu()

            # where are the correct answers?
            total_possible = logits.shape[1]
            locations = torch.argwhere(
                            torch.argsort(logits, descending=True) == labels.view(-1, 1)
                                        )[:, 1]

            ranks.append(locations)

    ranks = torch.concat(ranks)
    return {'accuracy': (ranks == 0).float().mean().item(),
            'identifiability': (1. - ranks.float() / total_possible).mean().item()
            }
