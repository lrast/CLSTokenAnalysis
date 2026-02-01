# model wrapper class for generalizable evaluations
from numpy import rad2deg
import torch
import json

from transformers import AutoModelForImageClassification, ViTForImageClassification
from src.utilities import HookContext


class ModelWrapper():
    """ Wrapper around models to handle all edge cases and provide a 
    standardized interface. Analyses should take a ModelWrapper.
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

    def add_internal_readouts(self, readout_functions, randomizers=None,
                              readout_method='straddle'):
        """ Initialize readout functions, hooks and trackers for multiple
        readouts

        readout_functions, randomizers: dictionaries indexed by layer names
        readout_method: how are the readouts attached to the original model?
        """
        hook_adders = {'CLS_token': self.add_CLS_activity_hook,
                       'straddle': self.add_straddle_hook
                       }

        add_hook = hook_adders[readout_method]

        tracked_layers = readout_functions.keys()
        randomizers = randomizers or {layer: None for layer in tracked_layers}

        # create dictionary to track outputs
        self.cached_activity = {layer: None for layer in tracked_layers}

        # initialize hooks
        handles = {}
        for layer in tracked_layers:
            handles[layer] = add_hook(layer, readout_functions[layer],
                                      randomizers[layer])

        self.readouts = readout_functions

        return HookContext(handles)

    def add_CLS_activity_hook(self, name, readout, randomizer):
        """Hook that applies readout to the CLS token activity and records the result
        """
        def record_readout_and_randomize(module, input, outputs):
            tuple_outs = isinstance(outputs, tuple)
            if tuple_outs:
                outputs = outputs[0]

            outputs = outputs.detach().clone()

            # record readouts
            cls_tokens = outputs[:, 0, :].cpu()
            with torch.no_grad():
                self.cached_activity[name] = readout(cls_tokens).cpu()

            if randomizer is None:
                to_return = outputs
            else:
                to_return = randomizer(outputs)

            if tuple_outs:
                return (to_return,)
            return to_return

        hook_handle = self.module_dict[name].register_forward_hook(
                                                    record_readout_and_randomize
                                                )

        return hook_handle

    def add_straddle_hook(self, name, readout, randomizer):
        """Hook in a model that straddles the module, modifying inputs
        and probing outputs
        """
        def record_readout_and_randomize(module, inputs, outputs):
            # track whether we are in the hook to prevent infinite loop on hook calls
            if module._in_hook:
                return

            module._in_hook = True
            try:
                if isinstance(inputs, tuple):
                    inputs = inputs[0]

                inputs = inputs.detach().clone()

                # record readouts
                with torch.no_grad():
                    self.cached_activity[name] = readout.forward(inputs, module)

                if randomizer is None:
                    return None
                else:
                    if isinstance(outputs, tuple):
                        return (randomizer(outputs[0]),)
                    else:
                        return outputs
            finally:
                module._in_hook = False

        self.module_dict[name]._in_hook = False
        hook_handle = self.module_dict[name].register_forward_hook(
                                                    record_readout_and_randomize
                                                )

        return hook_handle

    def get_batch_readout(self):
        """ Returns batch read-outs and resets the tracker for safety
        """
        outputs = self.cached_activity.copy()

        for layer in self.cached_activity.keys():
            self.cached_activity[layer] = None

        return outputs

    def zero_out_auxiliary_outputs(self):
        """Handle zeroing out non-CLS inputs to the final output layer in
        a model dependent way
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

    def to(self, device):
        """Moves model and probes to the specified device"""
        self.model = self.model.to(device)
        if hasattr(self, 'readouts'):
            for name, model in self.readouts.items():
                model.to(device)


# utilities for modelling
def shuffle_randomizer(batch):
    """CLS randomization by shuffling within batch"""
    ind_perm = torch.randperm(batch.shape[0])
    batch[:, 0, :] = batch[ind_perm, 0, :]

    return batch
