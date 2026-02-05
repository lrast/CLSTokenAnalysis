# Generate activity datasets for as sepcified model and dataset
import torch

from datasets import Dataset, IterableDataset, DatasetDict, load_from_disk
from torch.utils.data import DataLoader

from src.utilities import HookContext
from pathlib import Path


def activity_generator(model=None, datasplit=None, recorder=None, batch_size=64):
    """Activity generator: iterates through the dataset yields recorded values
    """
    dl = DataLoader(datasplit, batch_size=batch_size,
                    num_workers=min(4, datasplit.num_shards))

    with torch.no_grad():
        for batch in dl:
            labels = batch['label']
            inputs = batch['input'].to(model.device)

            _ = model(inputs)
            activity = recorder.pop()
            activity['label'] = labels.clone()

            keys = activity.keys()
            for values in zip(*activity.values()):
                yield dict(zip(keys, values))


def generate_activity_dataset(model_wrapper, dataset_dict, splits=['train', 'test'],
                              output_dir='temp_activity_dataset',
                              include_classifier_inputs=True,
                              batch_size=64, device=None, shuffle=False, seed=42,
                              ):
    """Generates a huggingface dataset on disk containing the CLS token activity
    at specified layers

    include_classifier_inputs: records the input values to the classifier for
    output probing
    """
    model = model_wrapper.model.to(device)
    recorder = ActivityRecorder(model_wrapper, include_classifier_inputs)

    full_dataset = DatasetDict()
    with recorder:
        for i, split in enumerate(splits):
            # write the results to a dataset
            datasplit = dataset_dict[split]
            data_subset = Dataset.from_generator(activity_generator, keep_in_memory=True,
                                                 gen_kwargs={'model': model,
                                                             'datasplit': datasplit,
                                                             'recorder': recorder,
                                                             'batch_size': batch_size
                                                             },
                                                 split=split)

            data_subset = data_subset.shuffle(seed=seed)
            full_dataset[split] = data_subset

    full_dataset.save_to_disk(output_dir)

    return load_from_disk(output_dir)


class ActivityRecorder(HookContext):
    """Records activity across the model"""
    def __init__(self, model_wrapper, include_classifier_inputs=True):
        super().__init__(hook_handles={})
        self.CLS_tokens = {}

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

    def pop(self):
        """Return CLS_tokens, set local versions to None"""
        outputs = self.CLS_tokens.copy()

        for key in self.CLS_tokens:
            self.CLS_tokens[key] = None

        return outputs


def generate_layer_input_datase(model_wrapper, layer_name, dataset_dict,
                                splits=['train', 'test'],
                                output_dir='temp_full_activity_dataset',
                                batch_size=64, device=None, shuffle=False,
                                seed=42):
    """Makes a dataset of all token activity for a single layer
    """
    model = model_wrapper.model.to(device)
    recorder = LayerInputHooks(model_wrapper, layer_name)

    with recorder:
        for i, split in enumerate(splits):
            # write the results to a dataset
            datasplit = dataset_dict[split]
            data_subset = Dataset.from_generator(activity_generator, keep_in_memory=True,
                                                 gen_kwargs={'model': model,
                                                             'datasplit': datasplit,
                                                             'recorder': recorder,
                                                             'batch_size': batch_size
                                                             },
                                                 split=split)

            data_subset = data_subset.shuffle(seed=seed)
            data_subset.save_to_disk(Path(output_dir) / split)

    return load_from_disk(output_dir)


class OnlineLayerInputDataset(IterableDataset):
    """
    Iterable dataset that yields {layer_name: input_activities, label: targets}
    from model layers, computed on the fly rather than stored to disk.
    """
    def __init__(self, model_wrapper, layer_names, datasplit, batch_size=64,
                 device=None):
        self.model_wrapper = model_wrapper
        self.layer_names = layer_names
        self.datasplit = datasplit
        self.batch_size = batch_size
        self.device = device

        self.model = model_wrapper.model.to(device)

        self._state_dict = {}

    def __iter__(self):
        dataloader = torch.utils.data.DataLoader(
            self.datasplit,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=getattr(self.datasplit, 'collate_fn', None)
        )
        recorder = LayerInputHooks(self.model_wrapper, self.layer_names)

        with recorder, torch.no_grad():
            for batch in dataloader:
                labels = batch['label']
                inputs = batch['input'].to(self.device)

                # Forward pass to populate input activity
                _ = self.model(inputs)
                return_values = recorder.pop()
                return_values['label'] = labels
                yield return_values

    def __len__(self):
        return (len(self.datasplit) + self.batch_size - 1) // self.batch_size


class LayerInputHooks(HookContext):
    """Records activity across the model"""
    def __init__(self, model_wrapper, layer_names):
        super().__init__(hook_handles={})

        if isinstance(layer_names, str):
            layer_names = [layer_names]

        self.input_activity = {}

        for layer_name in layer_names:
            module = model_wrapper.module_dict[layer_name]

            hook = module.register_forward_hook(
                                                self.create_input_hook(layer_name)
                                                )

            self.hook_handles[layer_name] = hook
            self.input_activity[layer_name] = None

    def create_input_hook(self, name):
        def recording_hook(module, inputs, output):
            if isinstance(inputs, tuple):
                # this can be tensor or tuple
                inputs = inputs[0]

            self.input_activity[name] = inputs.detach().clone().cpu()

        return recording_hook

    def pop(self):
        """Return activity, set local versions to None"""
        outputs = self.input_activity.copy()

        for key in self.input_activity:
            self.input_activity[key] = None

        return outputs
