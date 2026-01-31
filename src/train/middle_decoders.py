# training steps for intermediate decoders

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor


def train_module_decoder(decoder_model, base_module,
                         train_dataloader, val_dataloader,
                         max_epochs=3, patience=5,
                         accelerator="auto", devices="auto",
                         **trainer_kwargs):
    """
    Trains a ModuleSpecificDecoder using pytorch_lightning. Early stops on validation loss.
    Loads the best model weights at the end.

    Args:
        decoder_model: An ModuleSpecificDecoder instance
        base_module: The module that it is decoding
        train_dataloader: PyTorch DataLoader for training set.
        val_dataloader: PyTorch DataLoader for validation set.
        max_epochs: Maximum epochs to train.
        patience: Patience for early stopping on val_loss.
        accelerator: Accelerator for training ("cpu", "gpu", "auto", etc.).
        devices: Devices specification ("auto", int, etc.).
        **trainer_kwargs: Additional kwargs for Trainer.
    Returns:
        Trained (best weights) model.
    """

    # Early stopping on validation loss
    early_stop_cb = EarlyStopping(
        monitor="val/accuracy",
        patience=patience,
        verbose=True,
        mode="max"
    )

    # ModelCheckpoint to save the best model
    checkpoint_cb = ModelCheckpoint(
        monitor="val/accuracy",
        save_top_k=1,
        mode="max",
        save_last=True,
        filename="best"
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    wandb_logger = WandbLogger(project="middle_decoders", log_model=True)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        callbacks=[early_stop_cb, checkpoint_cb, lr_monitor],
        accelerator=accelerator,
        devices=devices,
        **trainer_kwargs,
    )

    trainer_model = TrainingWrapper_Decoder(decoder_model, base_module)

    trainer.fit(trainer_model, train_dataloader, val_dataloader)

    # Load the best checkpoint, if available
    best_model_path = checkpoint_cb.best_model_path
    if best_model_path:
        decoder_model = type(decoder_model).load_from_checkpoint(best_model_path)

    return decoder_model


class TrainingWrapper_Decoder(pl.LightningModule):
    """Surrogate Module to use for training the Module decoders 

        Manages base_module dependence, training hyperparameters
    """
    def __init__(self, decoder=None, base_module=None, lr=1E-3):
        super().__init__()
        self.decoder = decoder
        self.base_module = base_module
        self.lr = lr

        # freeze the parameters of the base module
        for param in self.base_module.parameters():
            param.requires_grad = False

        self.base_module.eval()

    def forward(self, input_tokens):
        return self.decoder.forward(input_tokens, self.base_module)

    def training_step(self, batch, batch_idx):
        """
        Assumes batch = (input_tokens, targets)
        - input_tokens: (B, L, D) float tensor
        - targets: (B,) class indices
        """
        input_tokens, targets = batch
        logits = self.forward(input_tokens)
        loss = self.decoder.loss_fn(logits, targets)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_tokens, targets = batch
        logits = self.forward(input_tokens)
        loss = self.decoder.loss_fn(logits, targets)
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

    def on_save_checkpoint(self, checkpoint):
        """ Strip the base_module weights from the checkpoint """
        checkpoint['state_dict'] = self.decoder.state_dict()


def fit_probes_by_ridge_regression(model_wrapper, activity_dataset, inds,
                                   internal_dim=768, num_classes=1000):
    """ Use cross-validated ridge regression to fit linear probes to initialize
    each of the model classifiers
    Use these weights to initialize the model classifier.

    activity_dataset: dataset of internal activity created by generate_activity_dataset
    """
    activity_dataset.set_format('pt')

    layer_names = [model_wrapper.layers[i] for i in inds]
    probes = {}

    for layer in layer_names:
        probe = torch.nn.Linear(internal_dim, num_classes)

        train_activity = activity_dataset['train'][:][layer]
        train_labels = activity_dataset['train'][:]['label']

        # fit SVM model
        encoder = OneHotEncoder()
        targets = 2*encoder.fit_transform(train_labels.reshape(-1, 1)).toarray() - 1

        regressor = Ridge()
        best_reg_model = RandomizedSearchCV(
                                    estimator=regressor, 
                                    param_distributions={'alpha': expon(scale=100)},
                                    n_iter=15,
                                    cv=5, 
                                    scoring='neg_mean_squared_error',
                                    refit=True
                                  )

        best_reg_model.fit(train_activity, targets)

        weights = best_reg_model.best_estimator_.coef_
        bias = best_reg_model.best_estimator_.intercept_

        with torch.no_grad():
            probe.weight.copy_(torch.as_tensor(weights))
            probe.bias.copy_(torch.as_tensor(bias))

        probes[layer] = probe

    return probes
