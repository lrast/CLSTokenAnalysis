# model wrapper class for generalizable evaluations
import torch
import json

import pytorch_lightning as pl

from transformers import AutoModelForImageClassification, ViTForImageClassification
from src.utilities import HookContext

from torch.optim.lr_scheduler import OneCycleLR
from dataclasses import dataclass

from lightning.pytorch.utilities import grad_norm


@dataclass
class TrainConfig:
    epochs: int = 5
    steps_per_epoch: int = 3125

    max_lr: float = 1e-3
    pct_start: float = 0.05
    weight_decay: float = 0.0
    backbone_eval: bool = True

    save_readout_only: bool = True
    readout_loss_weight: float = 0.0
    gradient_clip_val: float = None


class ModelWrapper(pl.LightningModule):
    """ Wrapper around models to handle all edge cases and provide a 
    standardized interface.

    - Analyses should take a ModelWrapper.

    - Also a parent object that all readouts can be trained an used from

    Arguments:
    model_id: backbone model
    num_classes: readout dimension
    readout_module: read-out attachment with setup, forward, and state_dict
        arguments
    train_cfg: a configuration dataclass as above
    """
    def __init__(self, model_id, num_classes, readout_module=None,
                 train_cfg: TrainConfig = TrainConfig()):
        super().__init__()

        self.model_id = model_id
        self.train_cfg = train_cfg

        self.save_hyperparameters()

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

        self.classification_loss = torch.nn.CrossEntropyLoss()

        # metadata setup
        with open('metadata.json', 'r') as f:
            metadata = json.load(f)['model'][model_id]

        self.module_dict = {k: v for k, v in self.model.named_modules()}

        self.layers = [metadata['template'].format(ind=i)
                       for i in range(metadata['max_ind'])]

        self.CLS_classifier_name = ('classifier' if "classifier_name" not in metadata
                                    else metadata["classifier_name"])

        self.add_readout(readout_module)

    def module_generator(self):
        for layer_name in self.layers:
            yield (layer_name, self.module_dict[layer_name])

    def get_classifier_module(self):
        return self.module_dict[self.CLS_classifier_name]

    def add_readout(self, readout):
        self.readout = readout
        if readout is not None:
            self.readout.setup([self.module_dict[l] for l in self.layers], 
                               internal_loss_weight=self.train_cfg.readout_loss_weight
                               )

    # core pytorch lightning functionality
    def forward(self, x, labels=None):
        """
        Computes the loss in this function if labels are passed
        """
        if self.readout is None:
            outs = self.model(x).logits
            if labels is not None:
                loss = self.classification_loss(outs, labels)
                return loss, outs
            else:
                return outs
        else:
            base_outs = self.model(x, output_hidden_states=True)
            return self.readout.forward(base_outs, labels=labels)

    def training_step(self, batch, batch_idx):
        """
        Train the model on a single batch.
        """
        inputs = batch['input']
        labels = batch['label']

        loss, outs = self.forward(inputs, labels=labels)

        # utility for additional logging from the readout.
        if self.readout is not None and hasattr(self.readout, 'to_log'):
            self.log_dict({f'train/{key}': value for key, value in
                          self.readout.to_log.items()})

        self.log('train/loss', loss)
        self.log('debug/min', outs.min().item())
        self.log('debug/max', outs.max().item())
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch['input']
        labels = batch['label']

        loss, outs = self.forward(inputs, labels=labels)

        preds = outs.argmax(dim=1)
        acc = (preds == labels).float().mean()

        self.log('val/loss', loss)
        self.log('val/accuracy', acc)
        return {'val_loss': loss, 'val_accuracy': acc}

    def test_step(self, batch, batch_idx):
        inputs = batch['input']
        labels = batch['label']

        outs = self.forward(inputs)
        preds = outs.argmax(dim=1)
        acc = (preds == labels).float().mean()

        self.log('test/accuracy', acc)
        return {'test_accuracy': acc}

    def on_before_optimizer_step(self, optimizer):
        """debug: gradient norm logging"""
        norms = grad_norm(self, norm_type='inf')
        self.log_dict(norms)

    def configure_optimizers(self):
        """Adam with cosine scheduling
        """
        lr = self.train_cfg.max_lr

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr / 25.)

        scheduler = OneCycleLR(optimizer, epochs=self.train_cfg.epochs,
                               steps_per_epoch=self.train_cfg.steps_per_epoch,
                               max_lr=lr, div_factor=25.0, final_div_factor=1e4,
                               anneal_strategy='cos', pct_start=self.train_cfg.pct_start)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_save_checkpoint(self, checkpoint):
        """Checkpointing behavior 
        """
        if self.train_cfg.save_readout_only:
            # don't save the backbone, only the readout
            if self.readout is None:
                base_dict = self.model.state_dict()

                state_dict = {k: v for k, v in base_dict.items()
                              if k.split('.')[0] == self.CLS_classifier_name}
            else:
                state_dict = self.readout.state_dict()

            checkpoint['state_dict'] = state_dict

    # utilities for enabling and disabling different parts of the model
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

    def freeze_backbone(self, freeze_classifier=False):
        """ freeze the backbone a put it into eval mode """
        for name, param in self.model.named_parameters():
            if ((not freeze_classifier) and
               (name.split('.')[0] == self.CLS_classifier_name)):
                # don't change anything inside the classifier
                pass
            else:
                param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()

    def train(self, mode=True):
        super().train(mode)
        if self.train_cfg.backbone_eval:  # keep backbone in eval while training
            self.model.eval()
        return self


# utilities for modelling
def shuffle_randomizer(batch):
    """CLS randomization by shuffling within batch"""
    ind_perm = torch.randperm(batch.shape[0])
    batch[:, 0, :] = batch[ind_perm, 0, :]

    return batch
