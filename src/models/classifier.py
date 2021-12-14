from typing import List
import os
import sys
import torch
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning import LightningModule
from transformers import BertModel
sys.path.append(os.path.join(os.getcwd(), 'src'))
# import logging
# logging.basicConfig(level=logging.INFO)


class Classifier(LightningModule):

    def __init__(self, params):
        super().__init__()
        self.criterion = CrossEntropyLoss()
        self.learning_rate = params['learning_rate']
        self.lr_update = params['lr_update']
        self.num_classes = params['num_classes']
        self.fine_tune = params['fine_tune'] # None if you encounter an error here when you load a model
        self.BertModel = BertModel.from_pretrained('bert-base-uncased')
        self.fc = Linear(768, self.num_classes)
        if not self.fine_tune:
            for param in self.BertModel.parameters():
                param.requires_grad = False
        self.save_hyperparameters()

    def forward(self, x):
        raise NotImplementedError("Forward pass needs to be implemented by specific model.")

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, idx):
        raise NotImplementedError

    def validation_epoch_end(self, outputs: List) -> None:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        log = {'val_loss': avg_loss, 'val_acc': avg_acc}
        self.log_dict(log, prog_bar=True, on_step=False, on_epoch=True)
        return

    def test_epoch_end(self, outputs: List) -> None:
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        log = {'test_loss': avg_loss, 'test_acc': avg_acc}
        self.log_dict(log, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=self.lr_update, gamma=0.1, verbose=True)  # learning rate decay
        return [optimizer], [scheduler]