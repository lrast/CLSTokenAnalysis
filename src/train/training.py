import evaluate
import torch
import wandb
import numpy as np

from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor


def train_readout(model, train_dataloader, val_dataloader, run_name, patience=5,
                  accelerator="auto", devices="auto",
                  **trainer_kwargs):
    """
    Trains a the readout portion of the model using pytorch_lightning.
    Loads the best model weights at the end.
    """

    model.freeze_backbone()
    train_cfg = model.train_cfg

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

    wandb_logger = WandbLogger(project="middle_decoders")

    trainer = pl.Trainer(
        max_epochs=train_cfg.epochs,
        logger=wandb_logger,
        callbacks=[early_stop_cb, checkpoint_cb, lr_monitor],
        accelerator=accelerator,
        devices=devices,
        **trainer_kwargs,
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    # Load the best checkpoint, save the final readout and the training configs
    best_model_path = checkpoint_cb.best_model_path
    ckpt = torch.load(best_model_path, weights_only=False)

    if model.readout is None:
        state_dict = {k: v for k, v in ckpt['state_dict'].items()
                      if k.split('.')[0] == model.CLS_classifier_name}

        model.model.save_pretrained(run_name + '/readout', state_dict=state_dict)

    else:
        model.readout.save_pretrained(run_name + '/readout',
                                      state_dict=ckpt['state_dict'])

    wandb.finish()


def run_probe_training(model, dataset, collator, **kwargs):
    """Probe trainer by SGD"""
    # Freeze all layers except the classification layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    # overridable arguments
    default_args = {
        'output_dir': "probe_train",
        'eval_strategy': "epoch",
        'save_strategy': "epoch",
        'learning_rate': 5e-5,
        'per_device_train_batch_size': 16,
        'gradient_accumulation_steps': 4,
        'num_train_epochs': 1,
        'logging_steps': 100,
    }
    settings = default_args | kwargs

    model = train_model(model, dataset, collator, **settings)

    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True 

    return model


def fit_probes_by_ridge_regression(model_wrapper, activity_dataset, **kwargs):
    """ Use cross-validated ridge regression to fit linear probes
    Use these weights to initialize the model classifier.

    activity_dataset: dataset of internal activity created by generate_activity_dataset
    """
    activity_dataset.set_format('pt')

    train_activity = activity_dataset['train'][:]['classifier_inputs']
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
        classifier_module = model_wrapper.get_classifier_module()
        classifier_module.weight.copy_(torch.as_tensor(weights))
        classifier_module.bias.copy_(torch.as_tensor(bias))

    return model_wrapper


def run_full_training(model, dataset, collator, **kwargs):
    """Full trainer"""

    # overridable arguments
    default_args = {
        'output_dir': "full_train",
        'eval_strategy': "epoch",
        'save_strategy': "epoch",
        'learning_rate': 5e-5,
        'per_device_train_batch_size': 16,
        'gradient_accumulation_steps': 4,
        'num_train_epochs': 3,
        'logging_steps': 100,
    }
    settings = default_args | kwargs

    model = train_model(model, dataset, collator, **settings)

    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True 

    return model


def run_lora_training(model, dataset, collator, **kwargs):
    """Lora trainer
    
        TO DO: Needs debugging

    """
    peft_config = LoraConfig(
                            r=16,
                            lora_alpha=16,
                            target_modules=["query", "value"],
                            lora_dropout=0.05,
                            modules_to_save=["classifier"],
                        )

    default_args = {
        'output_dir': "lora_train",
        'eval_strategy': "epoch",
        'save_strategy': "epoch",
        'learning_rate': 5e-5,
        'per_device_train_batch_size': 16,
        'gradient_accumulation_steps': 4,
        'num_train_epochs': 5,
        'logging_steps': 100,
    }
    settings = default_args | kwargs

    model = get_peft_model(model, peft_config)

    model = train_model(model, dataset, collator, **settings)

    return model


def train_model(model, dataset, collator, **kwargs):
    # setup accuracy metric
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
                        label_names=["label"],
                        remove_unused_columns=False, # for lazy preprocessing
                        load_best_model_at_end=True,
                        metric_for_best_model="accuracy",
                        report_to="wandb",
                        **kwargs
                    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return model
