# analysis utilities
import torch
import pandas as pd

from torch.utils.data import TensorDataset
from tqdm import tqdm

from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon


def run_across_layers(model, dataset, analysis, layers, **analysis_kwargs):
    """Runs an analysis for all layers in a model """
    results = []
    print(f'📕 Analysis: {analysis}')
    for layer_name in layers:
        print(f'📄 Layer: {layer_name}')

        stats = analysis(model, layer_name, dataset, **analysis_kwargs)
        results.append({'layer': layer_name} | stats)

    return pd.DataFrame(results)


# Layer randomization 
def accuracy_random_CLS(model, layer_name, dataset, randomization_mode='shuffle',
                        **evalkwargs):
    """ Randomize the class tokens in the specified layer, compute accuracy
    """
    if layer_name is not None:
        handle = add_randomization_hook(model, layer_name,
                                        randomization_mode=randomization_mode)

    outputs = model_accuracy(model, dataset["test"], **evalkwargs)

    if layer_name is not None:
        handle.remove()

    return {'accuracy': outputs, 'test': 'cls_randomization'}


def add_randomization_hook(model, layer_name, randomization_mode='shuffle'):
    """Randomizes CLS tokens """

    module_dict = {k: v for k, v in model.named_modules()}
    module = module_dict[layer_name]

    def randomization_hook(module, input, output):
        # another special case for tuple vs tensor
        is_tuple = isinstance(output, tuple)
        if is_tuple:
            copied_output = output[0].clone()
        else:
            copied_output = output.clone()

        batch, tokens, dims = copied_output.shape

        if randomization_mode == 'shuffle':
            ind_perm = torch.randperm(batch)
            copied_output[:, 0, :] = copied_output[ind_perm, 0, :]
        elif randomization_mode == 'normal':
            # default to normal
            copied_output[:, 0, :] = torch.randn(batch, dims)

        if is_tuple:
            return (copied_output,)

        return copied_output

    handle = module.register_forward_hook(randomization_hook)
    return handle


# Layer decoding
def linear_probe_by_ridge_regression(activity_dataset, cvfold=0, alpha=1.0):
    """
    Uses linear probing to assess class identity presence in the named layers
    - activity_dataset: a datase of layer-wise activity and labels
    """
    activity_dataset.set_format("pt")
    results = []

    for layer_name in activity_dataset['train'].column_names:
        if layer_name == 'label':
            continue

        train_embeddings = activity_dataset['train'][:][layer_name]
        test_embeddings = activity_dataset['test'][:][layer_name]

        train_labels = activity_dataset['train'][:]['label']
        test_labels = activity_dataset['test'][:]['label']

        encoder = OneHotEncoder()
        targets = 2*encoder.fit_transform(train_labels.reshape(-1, 1)).toarray() - 1

        if cvfold == 0:
            best_reg_model = Ridge(alpha=alpha)
        else:
            regressor = Ridge()
            best_reg_model = RandomizedSearchCV(
                                        estimator=regressor, 
                                        param_distributions={'alpha': expon(scale=100)},
                                        n_iter=15,
                                        cv=cvfold, 
                                        scoring='neg_mean_squared_error',
                                        refit=True
                                      )

        best_reg_model.fit(train_embeddings, targets)

        accuracy = (best_reg_model.predict(test_embeddings).argmax(1) == test_labels).float().mean().item()

        results.append({'layer': layer_name, 'accuracy': accuracy, 'test': 'layerwise_probe'})

    return pd.DataFrame(results)


def apply_model_decoder(model, activity_dataset, **evalkwargs):
    """ Uses the model's classifier to decode CLS tokens in given layer,
    Measuring accuracy and identifiability metrics
    """
    activity_dataset.set_format('pt')
    decoder = model.classifier

    results = []
    for layer_name in activity_dataset['test'].column_names:
        if layer_name == 'label':
            continue

        activity = activity_dataset['test'][:][layer_name]
        labels = activity_dataset['test'][:]['label']

        ds = TensorDataset(activity, labels)

        stats = classification_stats(decoder, ds, **evalkwargs)
        curr_results = {'layer': layer_name, 'test': 'model_decoder'} | stats
        results.append(curr_results)

    return pd.DataFrame(results)


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
            labels = batch['label']
            inputs = batch['input'].to(device)

            preds = model.forward(inputs).logits.argmax(1).cpu()
            outputs.append(preds.cpu() == labels)

    return torch.concat(outputs).float().mean().item()


def classification_stats(model, dataset, device=None, **evalkwargs):
    """ Accuracy and identifiability evaluation for raw pytorch models"""
    model = model.eval().to(device)

    if device is None:
        # robustly extract model device
        device = next(model.parameters()).device

    defaults = {'batch_size': 128, 'shuffle': False}
    evalkwargs = (defaults | evalkwargs)

    dl = torch.utils.data.DataLoader(dataset, **evalkwargs)
    ranks = []

    with torch.no_grad():
        for batch in iter(dl):
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


# Previous analysis: simultaneous randomization and readout
# This function need to be rewritten to account for better data creation.
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
