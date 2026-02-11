from src.utilities import HookContext

import torch


def test_hidden_states_vs_hooks(model_wrapper):
    """ Run a test to make ensure that the ith hidden state is the input the
    ith layer of the model
     """
    # test inputs
    for i, layer_name in enumerate(model_wrapper.layers):
        test_batch = torch.randn(64, 3, 224, 224, device=model_wrapper.model.device)

        with LayerInputHooks(model_wrapper, layer_name) as hook:
            outs = model_wrapper.model.forward(test_batch, output_hidden_states=True)
            outputs = outs.hidden_states[i].cpu()

            inputs = hook.pop()[layer_name].cpu()

            print(f'hidden state index: {i}, acts as inputs to layer {layer_name}: ', 
                  (outputs == inputs).all().item(),
                  f'\n hidden state count: {len(outs.hidden_states)}')

    # test outputs
    for i, layer_name in enumerate(model_wrapper.layers):
        i = i + 1  # for outputs
        test_batch = torch.randn(64, 3, 224, 224, device=model_wrapper.model.device)

        with LayerOutputHooks(model_wrapper, layer_name) as hook:
            outs = model_wrapper.model.forward(test_batch, output_hidden_states=True)
            outputs = outs.hidden_states[i].cpu()

            inputs = hook.pop()[layer_name].cpu()

            print(f'hidden state index: {i}, acts as outputs from layer {layer_name}: ', 
                  (outputs == inputs).all().item(),
                  f'\n hidden state count: {len(outs.hidden_states)}')


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


class LayerOutputHooks(HookContext):
    """Records activity across the model"""
    def __init__(self, model_wrapper, layer_names):
        super().__init__(hook_handles={})

        if isinstance(layer_names, str):
            layer_names = [layer_names]

        self.output_activity = {}

        for layer_name in layer_names:
            module = model_wrapper.module_dict[layer_name]

            hook = module.register_forward_hook(
                                                self.create_output_hook(layer_name)
                                                )

            self.hook_handles[layer_name] = hook
            self.output_activity[layer_name] = None

    def create_output_hook(self, name):
        def recording_hook(module, inputs, output):
            if isinstance(output, tuple):
                # this can be tensor or tuple
                output = output[0]

            self.output_activity[name] = output.detach().clone().cpu()

        return recording_hook

    def pop(self):
        """Return activity, set local versions to None"""
        outputs = self.output_activity.copy()

        for key in self.output_activity:
            self.output_activity[key] = None

        return outputs
