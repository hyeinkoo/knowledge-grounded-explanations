import os
import sys
import torch
from transformers import BertTokenizer
sys.path.append(os.path.join(os.getcwd(), 'src'))
from data.datamodule import DataModule
from models.multihop_kg import KnowledgeGraph


class KGDataset(DataModule):

    def __init__(self, args):
        super().__init__(args)
        # self.kg = KnowledgeGraph(params=self.params)

    def collate_fn(self, data):
        text = [x[0] for x in data]
        label = torch.tensor([int(x[1]) for x in data])
        return text, label


class Dataset(DataModule):

    def __init__(self, args):
        super().__init__(args)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_num_tokens = self.params['max_num_tokens']

    def collate_fn(self, data):
        text = torch.LongTensor([self.tokenizer.encode(x[0], padding='max_length', max_length=self.max_num_tokens) for x in data])
        label = torch.tensor([int(x[1]) for x in data])
        return text, label
