# analyze causality and decodability in a trained model
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import IterableDataset
from analysis import model_accuracy

from datasets import DatasetDict


def train_and_evaluate_decoder(base_model, layer_name, dataset, dims, **kwargs):
    """How well can we decode class identity from a given layer """
    decoder = Decoder(dims=dims)

    # Create datasets with shared hook using factory function
    embedding_datasets = create_cls_embedding_datasets(
        base_model=base_model,
        dataset_dict=dataset,
        layer_name=layer_name,
        **kwargs
    )

    dl_train = torch.utils.data.DataLoader(embedding_datasets['train'], shuffle=True)
    dl_eval = torch.utils.data.DataLoader(embedding_datasets['valid'], shuffle=False)

    checkpoint_callback = ModelCheckpoint(
        monitor='eval/accuracy',
        mode='max',
        save_top_k=1,
        filename='best-{epoch:02d}-{eval/accuracy:.3f}'
    )
    
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(decoder, dl_train, dl_eval)
    
    # Load the best checkpoint back into the decoder
    if checkpoint_callback.best_model_path:
        decoder = Decoder.load_from_checkpoint(checkpoint_callback.best_model_path)
    
    accuracy = model_accuracy(decoder, embedding_datasets['test'])
    return accuracy


class Decoder(pl.LightningModule):
    def __init__(self, dims, activation=nn.ReLU, lr=3e-4, dropout_p=0.0):
        """ MLP decoder for class.
        """
        super().__init__()
        self.save_hyperparameters()

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(activation())
                layers.append(nn.Dropout(p=dropout_p))

        self.model = nn.Sequential(*layers)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.log("train/accuracy", acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("eval/loss", loss, on_step=False, on_epoch=True)
        self.log("eval/accuracy", acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def create_cls_embedding_datasets(base_model, dataset_dict, layer_name,
                                  repeats=1, return_labels=True, batch_size=8, 
                                  seed=42, device=None, **kwargs):
    """Create train/valid/test CLSEmbeddingDataset instances sharing a single hook.
    Args:
        base_model: The base model to attach the hook to
        dataset_dict: Dictionary with keys 'train', 'valid', 'test' containing base datasets
        layer_name: Name of the layer to attach the hook to
        **kwargs: Additional arguments passed to CLSEmbeddingDataset
    
    Returns:
        Dictionary with keys 'train', 'valid', 'test' containing CLSEmbeddingDataset instances
    """
    # Create a single shared hook manager
    hook_manager = SharedHookManager(base_model, layer_name)
    
    # Create datasets sharing the same hook manager
    datasets = DatasetDict()
    for split in ['train', 'valid', 'test']:
        if split in dataset_dict:
            datasets[split] = CLSEmbeddingDataset(
                base_model=base_model,
                base_inputs=dataset_dict[split],
                layer_name=layer_name,
                repeats=repeats,
                return_labels=return_labels,
                batch_size=batch_size,
                shuffle=True if split == 'train' else False,
                seed=seed,
                device=device,
                hook_manager=hook_manager,
                **kwargs
            )
    
    return datasets, hook_manager


class SharedHookManager:
    """Manages a single forward hook shared across multiple datasets"""
    def __init__(self, base_model, layer_name):
        self.CLS_tokens = None
        module_dict = {k: v for k, v in base_model.named_modules()}
        self.hook_handle = module_dict[layer_name].register_forward_hook(self.recording_hook)
    
    def recording_hook(self, module, input, output):
        self.CLS_tokens = output[:, 0, :].detach().clone().cpu()
    
    def __del__(self):
        """Cleanup the hook"""
        del self.CLS_tokens
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()


class CLSEmbeddingDataset(IterableDataset):
    """Dateset of CLS token embeddings for elements of a base dataset
    """
    def __init__(self, base_model, base_inputs,
                 layer_name='embedding.vit.layernorm',
                 repeats=1, return_labels=True,
                 batch_size=8, shuffle=True, seed=42, device=None,
                 hook_manager=None
                 ):
        super(CLSEmbeddingDataset).__init__()

        self.base_model = base_model.to(device) if device else base_model
        self.base_inputs = base_inputs

        self.repeats = repeats
        self.return_labels = return_labels

        # Create a new hook manager if none provided
        if hook_manager is None:
            self.hook_manager = SharedHookManager(base_model, layer_name)
        else:
            self.hook_manager = hook_manager

        # Batching parameters
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        self.gen = torch.Generator()
        self.gen.manual_seed(seed)
    
    @property
    def CLS_tokens(self):
        """Access CLS_tokens from the hook manager"""
        return self.hook_manager.CLS_tokens

    def __iter__(self):
        try:
            if self.shuffle:
                inds = torch.randperm(len(self.base_inputs), generator=self.gen)
            else:
                inds = torch.arange(len(self.base_inputs))

            batches = [inds[i: i+self.batch_size].tolist() 
                       for i in range(0, len(self.base_inputs), self.batch_size)
                       ]

            for batch_inds in batches:
                images, labels = self.base_inputs[batch_inds]
                if self.repeats == 1:
                    preds = self.base_model.forward(images.to(self.device)).logits.argmax(1)
                    activity = self.CLS_tokens

                else:  # loop through the images individually, batching each repeat
                    preds = []
                    activity = []

                    for i in range(images.shape[0]):
                        repeated = images[i: i+1].expand(self.repeats, -1, -1, -1)

                        curr_preds = self.base_model.forward(repeated.to(self.device)
                                                             ).logits.argmax(1)

                        activity.append(self.CLS_tokens[None, :])
                        preds.append(curr_preds[None, :])

                    activity = torch.cat(activity)
                    preds = torch.cat(preds)

                if self.return_labels:
                    yield activity, labels.cpu(), preds.detach().cpu()
                else:
                    yield activity
        finally:
            # Safety: reset CLS_tokens to None at the end of iteration
            self.hook_manager.CLS_tokens = None
