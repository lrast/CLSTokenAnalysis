# Generate activity datasets for as sepcified model and dataset
import torch
from datasets import Dataset, DatasetDict, load_from_disk

from torch.utils.data import DataLoader


def generate_activity_dataset(model_wrapper, dataset_dict, splits=['train', 'test'],
                              output_dir='temp_activity_dataset',
                              include_classifier_inputs=True,
                              batch_size=64, device=None, shuffle=False, seed=42):
    """Generates a huggingface dataset on disk containing the CLS token activity
    at specified layers

    include_classifier_inputs: records the input values to the classifier for
    output probing
    """
    if device:
        model = model_wrapper.model.to(device)

    recorder = ActivityRecorder(model_wrapper, include_classifier_inputs)

    def activity_generator(split):
        ds = dataset_dict[split]
        dl = DataLoader(ds, batch_size=batch_size, num_workers=min(4, ds.num_shards))

        with torch.no_grad():
            for batch in dl:
                labels = batch['label']
                inputs = batch['input'].to(model.device)

                _ = model(inputs)
                activity = recorder.CLS_tokens.copy()
                activity['label'] = labels.clone()

                keys = activity.keys()
                for values in zip(*activity.values()):
                    yield dict(zip(keys, values))

    full_dataset = DatasetDict()
    for i, split in enumerate(splits):
        # write the results to a dataset
        data_subset = Dataset.from_generator(activity_generator, keep_in_memory=True,
                                             gen_kwargs={'split': split}, split=split)

        data_subset = data_subset.shuffle(seed=seed)
        full_dataset[split] = data_subset

    full_dataset.save_to_disk(output_dir)

    return load_from_disk(output_dir)


class ActivityRecorder:
    """Records activity across the model"""
    def __init__(self, model_wrapper, include_classifier_inputs=True):

        self.CLS_tokens = {}
        self.hook_handles = {}

        for layer_name, module in model_wrapper.module_generator():
            hook = module.register_forward_hook(
                                                self.create_output_hook(layer_name)
                                            )
            self.hook_handles[layer_name] = hook
            self.CLS_tokens[layer_name] = None

        if include_classifier_inputs:
            classifier_module = model_wrapper.get_classifier_module()
            hook = classifier_module .register_forward_hook(
                                        self.create_input_hook('classifier_inputs')
                                    )
            self.hook_handles['classifier_inputs'] = hook
            self.CLS_tokens['classifier_inputs'] = None

    def create_output_hook(self, name):
        def recording_hook(module, input, output):
            if isinstance(output, tuple):
                # this can be tensor or tuple
                output = output[0]

            self.CLS_tokens[name] = output[:, 0, :].detach().clone().cpu()

        return recording_hook

    def create_input_hook(self, name):
        def recording_hook(module, input, output):
            self.CLS_tokens[name] = input[0].detach().clone().cpu()

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
