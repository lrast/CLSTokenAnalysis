# Decoding models for specific layers, using random input tokens

import torch
from torch import nn
import pytorch_lightning as pl


class FrozenCLSTokenProbe(pl.LightningModule):
    """
    Pytorch lightning module that:
    1. Generates CLS tokens using a CLS generator.
    2. Replaces CLS tokens in input, and runs them through a frozen module
    3. Uses a linear probe to classify the outputs.
    """
    def __init__(
        self,
        base_module: nn.Module,             # The model layer to probe
        output_dim: int = 1000,       # Number of output classes 
        token_dim: int = 768,               # Dimension of individual tokens
        cls_token_idx: int = 0,             # Position of CLS token in sequence
        freeze_base_module: bool = True,  # Whether to freeze the external model's params
        lr: float = 1e-3
    ):
        super().__init__()

        self.base_module = [base_module]  # wrapped in a list to prevent it from being saved
        self.cls_generator = CLSGenerator()
        self.probe = nn.Linear(token_dim, output_dim)

        self.output_dim = output_dim
        self.cls_token_idx = cls_token_idx
        self.lr = lr

        if freeze_base_module:
            for param in self.base_module[0].parameters():
                param.requires_grad = False
            self.base_module[0].eval()

        self.save_hyperparameters(ignore=['base_module'])

        self.loss_fn = nn.CrossEntropyLoss()
    
    def replace_cls_token(self, input_tokens, new_cls_token):
        """
        Replace the CLS token(s) in input_tokens (B, L, D) at cls_token_idx with new_cls_token (B, D)
        """
        input_tokens = input_tokens.clone()
        input_tokens[:, self.cls_token_idx, :] = new_cls_token
        return input_tokens

    def forward(self, input_tokens):
        batch_size = input_tokens.shape[0]
        new_CLS = self.cls_generator.generate(batch_size)  # Shape: (B, D)

        modified = self.replace_cls_token(input_tokens, new_CLS)
        with torch.no_grad():
            out = self.base_module[0](modified)
        if isinstance(out, tuple):  # Handle models that return (logits, ...)
            out = out[0]

        cls_embedded = out[:, self.cls_token_idx, :]

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
        return torch.optim.Adam(self.parameters(), lr=self.lr)


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
