# Decoding models for specific layers, using random input tokens

import torch
import pytorch_lightning as pl

from torch import nn
from huggingface_hub import PyTorchModelHubMixin


class SimpleReadOutAttachment(pl.LightningModule, PyTorchModelHubMixin):
    """Class in charge of attaching internal readouts to the backbone model
    Single layer readout with no additional loss
    """
    def __init__(self, layer_ind, train_probe=True, **decoderKwargs):
        super().__init__()
        self.layer_ind = layer_ind
        self.decoder = ModuleSpecificDecoder(**decoderKwargs)
        self.classification_loss = torch.nn.CrossEntropyLoss()

        if not train_probe:
            for parameter in self.decoder.probe.parameters():
                parameter.requires_grad = False

    def setup(self, parent_layers, internal_loss_weight=0.0):
        self.internal_loss_weight = internal_loss_weight
        self.base_layer = parent_layers[self.layer_ind]

    def forward(self, hidden_states, labels=None):
        model_input = hidden_states[self.layer_ind]
        readouts = self.decoder.forward(model_input, self.base_layer)

        if labels is None:
            return readouts

        classification_loss = self.classification_loss(readouts, labels)
        return classification_loss, readouts

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """ Do not include
        """
        decoder_dict = self.decoder.state_dict(destination=destination, prefix=prefix,
                                               keep_vars=keep_vars)

        return {'decoder.' + k: v for k, v in decoder_dict.items()}


class SelfCalibratingReadout(pl.LightningModule, PyTorchModelHubMixin):
    """Class in charge of attaching internal readouts to the backbone model
    Addition loss for mis-calibration in predictions

    TODO: Needs to be implemented 
    """
    pass


class ModuleSpecificDecoder(pl.LightningModule, PyTorchModelHubMixin):
    """
    Pytorch lightning module that:
    1. Generates CLS tokens using a CLS generator.
    2. Replaces CLS tokens in input, and runs them through a frozen module
    3. Uses a linear probe to classify the outputs.
    """
    def __init__(self, output_dim=1000, token_dim=768, cls_token_idx=0,
                 mode='replace', generator_class=None):
        super().__init__()
        if generator_class is None:  # default that is correctly hoisted
            generator_class = CLSGenerator

        self.cls_generator = generator_class()
        self.probe = nn.Linear(token_dim, output_dim)
        self.cls_token_idx = cls_token_idx
        self.mode = mode

        self.loss_fn = nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def replace_cls_token(self, input_tokens, new_cls_token):
        """
        Replace the CLS token(s) in input_tokens (B, L, D) at cls_token_idx
        with new_cls_token (B, D)
        """
        input_tokens = input_tokens.clone()

        match self.mode:
            case 'null':
                return input_tokens
            case 'replace':
                input_tokens[:, self.cls_token_idx, :] = new_cls_token
                return input_tokens
            case 'augment':
                if self.cls_token_idx != 0:
                    raise NotImplementedError('Augmentation at non-zero index')
                new_tokens = torch.concat([new_cls_token[:, None, :], input_tokens],
                                          dim=1)
                return new_tokens

    def forward(self, input_tokens, base_module):
        new_CLS_tokens = self.cls_generator.generate(input_tokens.shape[0])

        modified = self.replace_cls_token(input_tokens, new_CLS_tokens)

        outputs = base_module(modified)
        if isinstance(outputs, tuple):  # Handle models that return (logits, ...)
            outputs = outputs[0]

        cls_embedded = outputs[:, self.cls_token_idx, :]

        logits = self.probe(cls_embedded)
        return logits


class MultiModuleDecoder(pl.LightningModule, PyTorchModelHubMixin):
    """
    Pytorch lightning module that:
    1. Generates CLS tokens for multiple layers using a CLS generator.
    2. Replaces CLS tokens in input, and runs them through frozen modules
    3. Uses a linear probe across layers to classify the outputs.
    """
    def __init__(self, num_layers=4, output_dim=1000, token_dim=768, cls_token_idx=0):
        super().__init__()
        self.cls_generators = nn.ModuleList([CLSGenerator() for i in range(num_layers)])
        self.probe = MLPProbe(num_layers*token_dim, num_layers*token_dim, output_dim)
        self.cls_token_idx = cls_token_idx

        self.loss_fn = nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def replace_cls_token(self, input_tokens, new_cls_token):
        """
        Replace the CLS token(s) in input_tokens (B, L, D) at cls_token_idx
        with new_cls_token (B, D)
        """
        input_tokens = input_tokens.clone()
        input_tokens[:, self.cls_token_idx, :] = new_cls_token
        return input_tokens

    def forward(self, inputs):
        """Inputs is a dictionary of token, module pairs
        """
        # sort by layer name for consistency
        layers = list(inputs.keys())
        layers.sort()

        all_CLS_tokens = []

        for i, layer in enumerate(layers):
            input_tokens, base_module = inputs[layer]

            new_CLS_tokens = self.cls_generators[i].generate(input_tokens.shape[0])
            modified = self.replace_cls_token(input_tokens, new_CLS_tokens)

            outputs = base_module(modified)
            if isinstance(outputs, tuple):  # Handle models that return (logits, ...)
                outputs = outputs[0]

            cls_embedded = outputs[:, self.cls_token_idx, :]
            all_CLS_tokens.append(cls_embedded)

        cls_embedded = torch.concat(all_CLS_tokens, dim=1)

        logits = self.probe(cls_embedded)
        return logits


class CLSGenerator(pl.LightningModule):
    def __init__(self, sample_dim=768, hidden_dim=512, num_layers=3,
                 output_dim=768):
        super().__init__()
        self.save_hyperparameters()

        layers = []
        input_dim = sample_dim
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    def generate(self, batch_size):
        device = next(self.parameters()).device
        initial = torch.randn(batch_size, self.hparams.sample_dim, device=device)

        return self.forward(initial)


class DeterministicCLSGenerator(pl.LightningModule):
    def __init__(self, sample_dim=768) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.token = torch.nn.Parameter(torch.randn(sample_dim))

    def generate(self, batch_size):
        return self.token.repeat(batch_size, 1)


class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=1000, num_layers=2,
                 activation=nn.ReLU):
        """
        MLPProbe is a multi-layer perceptron used as a probe
        Args:
            input_dim (int): Size of each input sample.
            hidden_dim (int): Size of hidden layers.
            output_dim (int): Size of the output layer (e.g., number of classes).
            num_layers (int): Number of hidden layers.
            activation (nn.Module): Activation function class to use between layers.
        """
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
