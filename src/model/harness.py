# model wrapper class for generalizable evaluations
import torch
import json

from transformers import AutoModelForImageClassification, ViTForImageClassification
from src.utilities import HookContext


class ModelWrapper():
    """ Wrapper around models to handle all edge cases and provide a 
    standardized interface. Analyses should take a ModelWrapper.
    """
    def __init__(self, model_id, num_classes):
        self.model_id = model_id

        # handle constructor edge cases
        constructor = AutoModelForImageClassification
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

        # metadata setup
        with open('metadata.json', 'r') as f:
            metadata = json.load(f)['model'][model_id]

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
        """Handle zeroing out non-CLS inputs to the final output layer in
        a model dependent way.

        Used for experiments that probe CLS readouts alone
        """
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

        return HookContext({'zero_aux': hook_handle})

    # device management functions for the wrapper

    def device(self):
        return self.model.device

    def to(self, device):
        """Moves model and probes to the specified device"""
        self.model = self.model.to(device)
        if hasattr(self, 'readouts'):
            for name, model in self.readouts.items():
                model.to(device)

        return self

    def forward_with_readouts(self, layers_to_readout, readout_inputs=True,
                              return_readouts=True):
        """Intermediate readouts from the model. Purely passive, no hooks
        """
        pass


# utilities for modelling
def shuffle_randomizer(batch):
    """CLS randomization by shuffling within batch"""
    ind_perm = torch.randperm(batch.shape[0])
    batch[:, 0, :] = batch[ind_perm, 0, :]

    return batch
