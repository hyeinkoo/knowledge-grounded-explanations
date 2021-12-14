# from typing import List
import os
import sys
import torch
from argparse import ArgumentParser
from pytorch_lightning import Trainer
sys.path.append(os.path.join(os.getcwd(), 'src'))
from utils.parameters import process_parameters_yaml
from train import load_data, load_model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='agnews')
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--gpus', default=1, type=int)
    args = parser.parse_args()
    params = process_parameters_yaml()

    args.model = args.model_path.split('/')[0]
    args.checkpoint = '/'.join(args.model_path.split('/')[1:])
    args.sample_data = None
    args.overfit = False
    params['num_classes'] = params[f'{args.dataset}_num_classes']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_data(args, 'test')
    model = load_model(args, params).to(device)
    model.eval()

    trainer = Trainer(gpus=args.gpus)
    trainer.test(model, datamodule=data)