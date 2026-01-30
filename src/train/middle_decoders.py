# training steps for intermediate decoders

import torch

from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


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


def train_frozen_cls_token_probe(
                                 model,
                                 train_dataloader,
                                 val_dataloader,
                                 max_epochs=3,
                                 patience=5,
                                 accelerator="auto",
                                 devices="auto",
                                 **trainer_kwargs
                                 ):
    """
    Trains a FrozenCLSTokenProbe using pytorch_lightning. Early stops on validation loss.
    Loads the best model weights at the end. Logs to wandb if a wandb_logger is provided.

    Args:
        model: An instance of FrozenCLSTokenProbe (must inherit from pl.LightningModule).
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

    wandb_logger = WandbLogger(project="middle_decoders", log_model=True)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        callbacks=[early_stop_cb, checkpoint_cb],
        accelerator=accelerator,
        devices=devices,
        **trainer_kwargs,
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    # Load the best checkpoint, if available
    best_model_path = checkpoint_cb.best_model_path
    if best_model_path:
        model = type(model).load_from_checkpoint(best_model_path)

    return model
