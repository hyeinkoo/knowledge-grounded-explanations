from typing import List
import os
import sys
import torch
sys.path.append(os.path.join(os.getcwd(), 'src'))
from models.classifier import Classifier
# import logging
# logging.basicConfig(level=logging.INFO)


class Bert(Classifier):

    def __init__(self, params):
        super().__init__(params)

    def forward(self, x):
        output = self.BertModel(input_ids=x)
        logits = self.fc(output['pooler_output'])
        return logits

    def training_step(self, batch, batch_idx):
        text, label = batch
        logits = self(text)
        loss = self.criterion(logits, label)
        _, pred = torch.max(logits, 1)
        accuracy = pred.eq(label).sum() / label.size(0)
        log = {'train_loss': loss, 'train_acc': accuracy}
        self.log_dict(log, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        text, label = batch
        logits = self(text)
        loss = self.criterion(logits, label)
        _, pred = torch.max(logits, 1)
        accuracy = pred.eq(label).sum() / label.size(0)
        log = {'val_loss': loss, 'val_acc': accuracy}
        # self.log_dict(log, prog_bar=False, on_step=False, on_epoch=True)
        return log

    def test_step(self, batch, idx):
        text, label = batch
        logits = self(text)
        loss = self.criterion(logits, label)
        _, pred = torch.max(logits, 1)
        accuracy = pred.eq(label).sum() / label.size(0)
        log = {'test_loss': loss, 'test_acc': accuracy}
        # self.log_dict(log, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return log
