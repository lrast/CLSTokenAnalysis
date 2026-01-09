# analyze causality and decodability in a trained model
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


from analysis import classification_stats

from analysis import create_cls_embedding_datasets


def train_and_evaluate_decoder(base_model, layer_name, dataset, dims, **embeddingKwargs):
    """Neural net decoder to assess how well can we decode class identity
    from a given layer.
    """
    decoder = Decoder(dims=dims)

    # Create datasets with shared hook using factory function
    embedding_datasets, hook_manager = create_cls_embedding_datasets(
        base_model=base_model,
        dataset_dict=dataset,
        layer_name=layer_name,
        **embeddingKwargs
    )

    with hook_manager:
        dl_train = torch.utils.data.DataLoader(embedding_datasets['train'], batch_size=None)
        dl_eval = torch.utils.data.DataLoader(embedding_datasets['valid'], batch_size=None)

        checkpoint_callback = ModelCheckpoint(
            monitor='eval/accuracy',
            mode='max',
            save_top_k=1,
            filename='best-{epoch:02d}-{eval/accuracy:.3f}'
        )
        
        # WandB logger for Lightning
        wandb_logger = WandbLogger(
            project="CLStokenInformation",
            name=f"decoder-{layer_name}",
            log_model=True,
        )
        
        trainer = pl.Trainer(
            max_epochs=10,
            callbacks=[checkpoint_callback],
            logger=wandb_logger,
            log_every_n_steps=10,
        )
        trainer.fit(decoder, dl_train, dl_eval)
        
        # Load the best checkpoint back into the decoder
        if checkpoint_callback.best_model_path:
            decoder = Decoder.load_from_checkpoint(checkpoint_callback.best_model_path)
        
        accuracy = classification_stats(decoder, embedding_datasets['test'],
                                        batch_size=None, shuffle=False)['accuracy']
        # Log test accuracy to WandB
        wandb_logger.log_metrics({"test/accuracy": accuracy})

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
        logits = self.forward(x)

        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.log("train/accuracy", acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)

        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("eval/loss", loss, on_step=False, on_epoch=True)
        self.log("eval/accuracy", acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
