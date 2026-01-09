from transformers import AutoImageProcessor, AutoModelForImageClassification, \
                        DefaultDataCollator
from datasets import load_dataset


def setup_model_ds_collator_images(model_id, dataset_id, **kwargs):
    """ Sets-up the model, dataset, and trainer for an image processing run
    """
    ds = load_dataset(dataset_id)

    model = AutoModelForImageClassification.from_pretrained(
        model_id, 
        num_labels=len(ds["train"].features["label"].names),
        ignore_mismatched_sizes=True
    )

    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)

    def transform_images(batch):
        inputs = processor([img.convert("RGB") for img in batch["image"]],
                           return_tensors="pt")
        inputs["labels"] = batch["label"]
        return inputs

    ds = ds.with_transform(transform_images)
    data_collator = DefaultDataCollator()

    return model, ds, data_collator


def setup_model_ds_collator_text(model_id, dataset_id):
    """ Sets-up the model, dataset, and trainer for text processing run
    """
    pass


def setup_generic(model_id, dataset_id):
    """ Router for different tasks """
    ds = load_dataset(dataset_id)

    if 'image' in ds['train'].features.keys(): # image dataset
        return setup_model_ds_collator_images(model_id, dataset_id)

    elif 'text' in ds['train'].features.keys():
        return setup_model_ds_collator_text(model_id, dataset_id)
