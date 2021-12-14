from typing import List
import os
import sys
import torch
sys.path.append(os.path.join(os.getcwd(), 'src'))
from models.classifier import Classifier
from models.multihop_kg import KnowledgeGraph
# import logging
# logging.basicConfig(level=logging.INFO)


class KBert(Classifier):

    def __init__(self, params):
        super().__init__(params)
        self.kg = KnowledgeGraph(params=params)
        self.use_visible_matrix = params['use_visible_matrix']

    def forward(self, token_id, mask, soft_idx, visible_matrix):
        if not self.use_visible_matrix:
            # TODO: integrate visible_matrix
            visible_matrix = None
        output = self.BertModel(input_ids=token_id, attention_mask=mask, position_ids=soft_idx)
        logits = self.fc(output['pooler_output'])
        return logits

    def training_step(self, batch, batch_idx):
        text, label = batch
        token_id, mask, soft_idx, visible_matrix, _, _ = self.kg.add_knowledge(text)
        token_id, mask, soft_idx, visible_matrix = self.set_device([token_id, mask, soft_idx, visible_matrix])
        logits = self(token_id, mask, soft_idx, visible_matrix)
        loss = self.criterion(logits, label)
        _, pred = torch.max(logits, 1)
        accuracy = pred.eq(label).sum() / label.size(0)
        log = {'train_loss': loss, 'train_acc': accuracy}
        self.log_dict(log, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        text, label = batch
        token_id, mask, soft_idx, visible_matrix, _, _ = self.kg.add_knowledge(text)
        token_id, mask, soft_idx, visible_matrix = self.set_device([token_id, mask, soft_idx, visible_matrix])
        logits = self(token_id, mask, soft_idx, visible_matrix)
        loss = self.criterion(logits, label)
        _, pred = torch.max(logits, 1)
        accuracy = pred.eq(label).sum() / label.size(0)
        log = {'val_loss': loss, 'val_acc': accuracy}
        # self.log_dict(log, prog_bar=False, on_step=False, on_epoch=True)
        return log

    def test_step(self, batch, idx):
        text, label = batch
        token_id, mask, soft_idx, visible_matrix, _, _ = self.kg.add_knowledge(text)
        token_id, mask, soft_idx, visible_matrix = self.set_device([token_id, mask, soft_idx, visible_matrix])
        logits = self(token_id, mask, soft_idx, visible_matrix)
        loss = self.criterion(logits, label)
        _, pred = torch.max(logits, 1)
        accuracy = pred.eq(label).sum() / label.size(0)
        log = {'test_loss': loss, 'test_acc': accuracy}
        # self.log_dict(log, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return log

    def set_device(self, data: List):
        for i, d in enumerate(data):
            data[i] = d.to(self.device)
        return data