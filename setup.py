from transformers import AutoImageProcessor, AutoModelForImageClassification, \
                         DefaultDataCollator
#                        AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

from datasets import load_dataset


def image_model_setup(model_id, dataset_id, num_classes):
    """ Sets-up the model, dataset, and trainer for an image processing run
    """
    ds = load_dataset(dataset_id, streaming=True)

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


# general picklable transformations
def transform_images(batch, processor=None, output_name="input"):
    """General parallalizable image transform"""
    outputs = processor([img.convert("RGB") for img in batch["image"]],
                        return_tensors=None)
    outputs[output_name] = outputs.pop('pixel_values') 
    if 'labels' in outputs:
        outputs['label'] = outputs.pop('labels')

    return outputs
