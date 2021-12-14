import os
import sys
import torch
from torch.utils.data import random_split, DataLoader#, RandomSampler
from pytorch_lightning import LightningDataModule
sys.path.append(os.path.join(os.getcwd(), 'src'))
from utils.loader import load_file
from utils.parameters import process_parameters_yaml


class DataModule(LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.params = process_parameters_yaml()
        self.dataset_path = self.params[f'{args.dataset}_dataset_path']
        self.train_file = self.params[f'{args.dataset}_processed_train_file']
        self.test_file = self.params[f'{args.dataset}_processed_test_file']
        self.batch_size = args.batch_size
        self.sample_data = args.sample_data
        self.overfit = args.overfit

    def prepare_data(self):
        # file path
        TRAIN_FILE_PATH = os.path.join(self.dataset_path, self.train_file)
        TEST_FILE_PATH = os.path.join(self.dataset_path, self.test_file)
        # reading files as csv
        self.train_data = load_file(TRAIN_FILE_PATH)
        self.test_data = load_file(TEST_FILE_PATH)

    def setup(self):
        if self.sample_data:
            rand_idx = torch.randperm(len(self.train_data))[:self.sample_data]
            train_idx, val_idx, test_idx = torch.split(rand_idx, [int(self.sample_data*0.6), int(self.sample_data*0.2), int(self.sample_data*0.2)])
            self.trainset = torch.utils.data.Subset(self.train_data, train_idx)
            self.valset = torch.utils.data.Subset(self.train_data, val_idx)
            self.testset = torch.utils.data.Subset(self.train_data, test_idx)
        else:
            threshold = round(len(self.train_data) * 0.8)

            self.trainset, self.valset = random_split(self.train_data,
                                                          [threshold, len(self.train_data) - threshold])
            self.testset = self.test_data

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=4, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=4, collate_fn=self.collate_fn)

    def test_dataloader(self):
        if self.overfit:
            return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=4, collate_fn=self.collate_fn)
        else:
            return DataLoader(self.testset, batch_size=self.batch_size, num_workers=4, collate_fn=self.collate_fn)

    def len_train(self):
        return len(self.trainset)

    def len_val(self):
        return len(self.valset)

    def len_test(self):
        if self.overfit:
            return len(self.trainset)
        else:
            return len(self.testset)

    def collate_fn(self):
        raise NotImplementedError