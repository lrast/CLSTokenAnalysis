import torch
import tempfile

from transformers import AutoImageProcessor, AutoModelForImageClassification, \
                        AutoTokenizer, AutoModelForSequenceClassification, \
                        DefaultDataCollator, DataCollatorWithPadding


from datasets import load_dataset, load_from_disk, DatasetDict, IterableDatasetDict, Dataset
from datasets.arrow_writer import ArrowWriter

from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
from pathlib import Path


def transform_images(batch, processor=None, output_name="input"):
    """General parallalizable image transform"""
    outputs = processor([img.convert("RGB") for img in batch["image"]],
                        return_tensors=None)
    outputs[output_name] = outputs.pop('pixel_values') 
    if 'labels' in outputs:
        outputs['label'] = outputs.pop('labels')

    return outputs


def image_model_setup(model_id, dataset_id, num_classes, on_disk=False, 
                      preshuffle=True, seed=42):
    """ Sets-up the model, dataset, and trainer for an image processing run
     - on_disk: Determines whether the dataset is fully downloaded or streamed
                this function will always return a streaming dataset
     - preshuffle: Saves and reloads a shuffled version of the dataset.
                   Only applicable when on_disk=True
    """
    ds = load_dataset(dataset_id, streaming=(not on_disk))

    if preshuffle and on_disk:
        # Preshuffle the dataset splits to ensure good samples,
        # Save and reload for performance
        for key in ds:
            ds[key] = ds[key].shuffle(seed=seed)

        ds.save_to_disk("tmp_shuffled_dataset", max_shard_size="500MB")
        ds = load_from_disk("tmp_shuffled_dataset")

    if on_disk:
        # convert to iterable
        ds = IterableDatasetDict({
             key: ds[key].to_iterable_dataset(num_shards=1024) for key in ds
            })

    model = AutoModelForImageClassification.from_pretrained(
        model_id, num_labels=num_classes, ignore_mismatched_sizes=True
    )

    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
    data_collator = DefaultDataCollator()

    ds = ds.map(
                transform_images, 
                batched=True, 
                fn_kwargs={"processor": processor, "output_name": "input"},
                remove_columns=["image"]
                )

    return model, ds, data_collator


def generate_activity_dataset(model, dataset_dict, layer_names, num_samples, 
                              dataset_keys=['train', 'test'],
                              data_dir='tmp_activity',
                              device=None, shuffle=True, seed=42, batch_size=64):
    """Generates a huggingface dataset on disk containing the CLS token activity
    at specified layers
    """
    if device:
        model = model.to(device)

    def make_dl(split):
        ds = dataset_dict[split]
        if shuffle:
            ds = ds.shuffle(seed=42, buffer_size=10_000)
        dl = DataLoader(ds.take(num_samples), batch_size=batch_size,
                        num_workers=min(4, ds.num_shards))
        return dl

    recorder = ActivityRecorder(model, layer_names)

    activity_ds = DatasetDict({})

    with tempfile.TemporaryDirectory() as tmpdir, torch.no_grad():
        for split in dataset_keys:
            dl = make_dl(split)
            data_file = Path(tmpdir) / f'data.arrow_{split}'

            with ArrowWriter(path=data_file) as writer:
                for batch in tqdm(dl, total=num_samples / batch_size):
                    labels = batch['label']
                    inputs = batch['input'].to(model.device)

                    _ = model(inputs)
                    activity = recorder.CLS_tokens.copy()
                    activity['label'] = labels
                    writer.write_batch(activity)

            activity_ds[split] = Dataset.from_file(str(data_file))

    activity_ds.save_to_disk("activity_dataset_temp")
    activity_ds = load_from_disk("activity_dataset_temp")

    return activity_ds


class ActivityRecorder:
    """Manages a single forward hook shared across multiple datasets"""
    def __init__(self, base_model, layer_names):
        module_dict = {k: v for k, v in base_model.named_modules()}
        self.CLS_tokens = {name: None for name in layer_names}
        self.hook_handles = {name: None for name in layer_names}

        for layer_name in layer_names:
            hook = module_dict[layer_name].register_forward_hook(
                                                self.create_hook(layer_name)
                                            )
            self.hook_handles[layer_name] = hook

    def create_hook(self, name):
        def recording_hook(module, input, output):
            self.CLS_tokens[name] = output[:, 0, :].detach().clone().cpu()

        return recording_hook

    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup the hook"""
        self.cleanup()
        return False  # Don't suppress exceptions

    def reset(self):
        for key in self.CLS_tokens:
            self.CLS_tokens[key] = None

    def cleanup(self):
        """Explicitly remove the hook. Should be called when done."""
        for name, handle in self.hook_handles.items():
            handle.remove()
            self.hook_handles[name] = None
        self.reset()

    def __del__(self):
        """Cleanup the hook as fallback"""
        self.cleanup()


# Older activity datasets: using lazy model evaluation

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

    def reset(self):
        self.CLS_tokens = None

    def cleanup(self):
        """Explicitly remove the hook. Should be called when done."""
        if hasattr(self, 'hook_handle') and self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
        self.reset()

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
