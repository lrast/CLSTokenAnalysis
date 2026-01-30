# Decoding models for specific layers, using random input tokens

import torch
import pytorch_lightning as pl

from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from huggingface_hub import PyTorchModelHubMixin


class ModuleSpecificDecoder(pl.LightningModule, PyTorchModelHubMixin):
    """
    Pytorch lightning module that:
    1. Generates CLS tokens using a CLS generator.
    2. Replaces CLS tokens in input, and runs them through a frozen module
    3. Uses a linear probe to classify the outputs.
    """
    def __init__(
        self,
        base_module: nn.Module,             # The model layer to probe
        output_dim: int = 1000,             # Number of output classes 
        token_dim: int = 768,               # Dimension of individual tokens
        cls_token_idx: int = 0,             # Position of CLS token in sequence
        lr: float = 1e-3,
        device: str = None
    ):
        super().__init__()

        self.base_module = [base_module]  # wrapped in a list to prevent saving
        self.cls_generator = CLSGenerator()
        self.probe = nn.Linear(token_dim, output_dim)

        self.output_dim = output_dim
        self.cls_token_idx = cls_token_idx
        self.lr = lr

        # freeze the parameters of the base module
        for param in self.base_module[0].parameters():
            param.requires_grad = False
        self.base_module[0].eval()

        self.save_hyperparameters(ignore=['base_module'])

        if device is not None:
            self.to(device)

        self.loss_fn = nn.CrossEntropyLoss()

    def replace_cls_token(self, input_tokens, new_cls_token):
        """
        Replace the CLS token(s) in input_tokens (B, L, D) at cls_token_idx
        with new_cls_token (B, D)
        """
        input_tokens = input_tokens.clone()
        input_tokens[:, self.cls_token_idx, :] = new_cls_token
        return input_tokens

    def forward(self, input_tokens):
        batch_size = input_tokens.shape[0]
        new_CLS = self.cls_generator.generate(batch_size)  # Shape: (B, D)

        modified = self.replace_cls_token(input_tokens, new_CLS)
        outputs = self.base_module[0](modified)
        if isinstance(outputs, tuple):  # Handle models that return (logits, ...)
            outputs = outputs[0]

        cls_embedded = outputs[:, self.cls_token_idx, :]

        logits = self.probe(cls_embedded)
        return logits

    def training_step(self, batch, batch_idx):
        """
        Assumes batch = (input_tokens, targets)
        - input_tokens: (B, L, D) float tensor
        - targets: (B,) class indices
        """
        input_tokens, targets = batch
        logits = self.forward(input_tokens)
        loss = self.loss_fn(logits, targets)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_tokens, targets = batch
        logits = self.forward(input_tokens)
        loss = self.loss_fn(logits, targets)
        acc = (logits.argmax(-1) == targets).float().mean()
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/accuracy", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # T_max is the number of epochs to reach the minimum LR
        scheduler = CosineAnnealingLR(optimizer, T_max=3*3125, eta_min=1e-6)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def to(self, device_name):
        """Also move the base module that we are conditioned on"""
        self.base_module[0].to(device_name)
        super().to(device_name)






class CLSGenerator(pl.LightningModule):
    def __init__(self, sample_dim=768, hidden_dim=512, num_layers=3,
                 output_dim=768):
        super().__init__()
        self.save_hyperparameters()

        layers = []
        input_dim = sample_dim
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
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
