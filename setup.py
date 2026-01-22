import json

from transformers import AutoImageProcessor, AutoModelForImageClassification, \
                         DefaultDataCollator, ViTForImageClassification

from datasets import load_dataset


def image_model_setup(model_id, dataset_id, num_classes):
    """ Sets-up the model, dataset, and trainer for an image processing run
    """
    ds = load_dataset(dataset_id, streaming=True)

    model = ModelWrapper(model_id, num_classes)

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
                        return_tensors='pt')
    outputs[output_name] = outputs.pop('pixel_values') 
    if 'labels' in outputs:
        outputs['label'] = outputs.pop('labels')

    return outputs


class ModelWrapper():
    """ Wrapper around models to handle all edge cases and provide a 
    standardized interface. Analyses should take a ModelWrapper
    """
    def __init__(self, model_id, num_classes):
        # model setup 
        self.model_id = model_id

        constructor = AutoModelForImageClassification  # handle constructor edge cases
        kwargs = {}
        if model_id == "facebook/vit-mae-base":
            constructor = ViTForImageClassification

        if model_id == "microsoft/beit-base-patch16-224-pt22k-ft22k":
            kwargs['use_mean_pooling'] = False

        self.model = constructor.from_pretrained(model_id,
                                                 num_labels=num_classes,
                                                 ignore_mismatched_sizes=True,
                                                 **kwargs
                                                 )

        with open('metadata.json', 'r') as f:
            metadata = json.load(f)['model'][model_id]

        # metadata setup
        self.module_dict = {k: v for k, v in self.model.named_modules()}

        self.layers = [metadata['template'].format(ind=i)
                       for i in range(metadata['max_ind'])]

        self.CLS_classifier_name = ('classifier' if "classifier_name" not in metadata
                                    else metadata["classifier_name"])

    def module_generator(self):
        for layer_name in self.layers:
            yield (layer_name, self.module_dict[layer_name])

    def get_classifier_module(self):
        return self.module_dict[self.CLS_classifier_name]

    def zero_out_auxiliary_outputs(self):
        """Handle zeroing out non-CLS inputs in a model dependent way"""
        if self.model_id == "facebook/dinov2-base":
            # dinov2 appends the CLS tokens and mean outputs as inputs to the classifier
            # The right approach in this case is to zero-out, fit probe, test
            def overwrite_avg(module, input):
                input = input[0]
                input[:, 768:] = 0.
                return (input,)

            hook_handle = self.module_dict[self.CLS_classifier_name
                                           ].register_forward_pre_hook(overwrite_avg)

        if self.model_id == "openai/clip-vit-base-patch32":
            # checking the implementation, this uses average-pooling only
            raise Exception("""This is an average-pooling only model""")

        if self.model_id == "facebook/deit-base-distilled-patch16-224":
            # forward hook on the whole model to use the cls_logits
            def redefine_outputs(module, input, outputs):
                outputs.logits = outputs.cls_logits
                return outputs

            hook_handle = self.module_dict[''].register_forward_hook(redefine_outputs)

        return HookContext(hook_handle)


class HookContext:
    """ Context that clears hook when it is done """
    def __init__(self, hook_handle):
        self.hook_handle = hook_handle

    def cleanup(self):
        """Explicitly remove the hook. Should be called when done."""
        self.hook_handle.remove()

    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup the hook"""
        self.hook_handle.remove()
        return False

    def __del__(self):
        """Cleanup the hook as fallback"""
        self.cleanup()
