import numpy as np
import evaluate

from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model


def run_probe_training(model, dataset, collator, **kwargs):
    """Probe trainer"""
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
                        label_names=["labels"], # for LoRA
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
