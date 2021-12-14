# from typing import List

import os
import sys
import torch
# import logging
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
sys.path.append(os.path.join(os.getcwd(), 'src'))
from utils.parameters import process_parameters_yaml
from data.dataloader import KGDataset, Dataset
from models.k_bert import KBert
from models.bert import Bert


def load_data(args, mode='train'):
    print(f"Loading Dataset: {args.dataset}")
    if args.model == 'kbert':
        data = KGDataset(args)
    elif args.model == 'bert':
        data = Dataset(args)
    else:
        raise Exception("No model")
    data.prepare_data()
    data.setup()
    if mode == 'train':
        print(f"Number of train samples: {data.len_train()}")
        print(f"Number of val samples:   {data.len_val()}")
    print(f"Number of test samples:  {data.len_test()}")
    return data


def load_model(args, params):
    if args.model == 'kbert':
        model = KBert
    elif args.model == 'bert':
        model = Bert
    else:
        raise Exception("No model")
    if args.checkpoint:
        print('Loading Checkpoint')
        checkpoint_path = f'checkpoints/{args.model}/{args.checkpoint}'
        model = model.load_from_checkpoint(checkpoint_path)
    else:
        print(f"Loading Model: {args.model}")
        model = model(params)
    return model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='agnews')#, choices=["agnews"])
    parser.add_argument('--model', default='kbert', choices=["kbert", "bert"])
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--lr', default=None, type=float)
    parser.add_argument('--lr_update', default=None, type=int)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--sample_data', default=None, type=int, help="size of smaller dataset")
    parser.add_argument('--overfit', default=False, action='store_true')
    args = parser.parse_args()
    params = process_parameters_yaml()

    if args.lr:
        params['learning_rate'] = args.lr
    if args.lr_update:
        params['lr_update'] = args.lr_update
    params['num_classes'] = params[f'{args.dataset}_num_classes']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_data(args)

    model = load_model(args, params).to(device)

    experiment = f'batch_size_{args.batch_size}-max_epoch{args.max_epochs}-sample_{args.sample_data}-overfit_{args.overfit}'
    logger = TensorBoardLogger(save_dir=f'logs/{args.model}', name=experiment)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath=f'checkpoints/{args.model}/{experiment}',
                                          filename='{epoch}-{step}-{val_loss:.2f}',
                                          every_n_val_epochs=1)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(gpus=args.gpus,
                      logger=logger,
                      max_epochs=args.max_epochs,
                      callbacks=[checkpoint_callback, lr_monitor])

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)
