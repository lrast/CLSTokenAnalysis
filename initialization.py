import torch
import tempfile

from transformers import AutoImageProcessor, AutoModelForImageClassification, \
                         DefaultDataCollator
#                        AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, Image

from datasets.arrow_writer import ArrowWriter

from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path


def image_model_setup(model_id, dataset_id, num_classes, samples_per_class=200,
                      splits=['train', 'test'], seed=42):
    """ Sets-up the model, dataset, and trainer for an image processing run
     - samples_per_class: stratified sampling of the dataset with this many samples
    """
    ds = load_dataset(dataset_id, streaming=True)

    if samples_per_class is not None:
        ds = stratified_image_subset(ds, num_classes, samples_per_class=samples_per_class,
                                     splits=splits)

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


def transform_images(batch, processor=None, output_name="input"):
    """General parallalizable image transform"""
    outputs = processor([img.convert("RGB") for img in batch["image"]],
                        return_tensors=None)
    outputs[output_name] = outputs.pop('pixel_values') 
    if 'labels' in outputs:
        outputs['label'] = outputs.pop('labels')

    return outputs


def stratified_image_subset(dataset_dict, num_classes, samples_per_class=200,
                            splits=['train', 'test'], stream_from_disk=True):
    """Stratified sub-sampling of the dataset to, producing a class-balanced
    split. This is the subsample that we need for most analyses.

    stream_from_disk: useful if later preprocessing steps are memory intensive
    """
    dataset_dict = dataset_dict.cast_column("image", Image(decode=False))

    def generate_samples(split=None):
        local_dataset = dataset_dict[split]

        remaining = {i: samples_per_class for i in range(num_classes)}
        total_to_find = num_classes * samples_per_class
        count_found = 0

        for batch in local_dataset:
            if count_found == total_to_find:
                break

            label = batch['label']

            if (count_found == 0) and (label not in remaining):
                # fail loundly if the first label is unknown
                raise ValueError('Unknown label value')

            if label in remaining:
                count_found += 1

                remaining[label] -= 1
                if remaining[label] == 0:
                    # remove it from the dict
                    del remaining[label]

                yield batch

    base_dir = Path('temp_dataset_subsample')
    for split in splits:
        # write the results to a dataset
        data_subset = Dataset.from_generator(generate_samples, keep_in_memory=True,
                                             gen_kwargs={'split': split},split=split)
        data_subset = data_subset.cast_column("image", Image(decode=True))

        data_subset.save_to_disk(base_dir / split)

    return load_dataset('arrow', data_dir=base_dir, streaming=True)


def generate_activity_dataset(model, dataset_dict, layer_names, num_samples, 
                              splits=['train', 'test'],
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
            ds = ds.shuffle(seed=seed, buffer_size=10_000)
        dl = DataLoader(ds.take(num_samples), batch_size=batch_size,
                        num_workers=min(4, ds.num_shards))
        return dl

    recorder = ActivityRecorder(model, layer_names)

    activity_ds = DatasetDict({})

    with tempfile.TemporaryDirectory() as tmpdir, torch.no_grad():
        for split in splits:
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

    activity_ds.save_to_disk("temp_activity_dataset")
    activity_ds = load_from_disk("temp_activity_dataset")

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
