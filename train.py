from typing_extensions import get_args
import torch
from attrdict import AttrDict
import argparse
import json

import dataloader.datasets
from training_loop import train

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Train ACTGAN models.'
    )
    parser.add_argument(
        '--config',
        help='path to training config json file',
        required=True
    )
    args = parser.parse_args()
    # Load config
    
    # Load config from file
    with open(args.config) as f:
        cfg = AttrDict(json.load(f))

    # Load dataset
    if cfg.dataset=='cifar':
        train_data= dataloader.datasets.get_cifar_data(cfg)
    elif cfg.dataset=='mnist':
        train_data = dataloader.datasets.get_mnist_data(cfg)
    
    # Train Loader
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True)

    # Run training loop
    train(cfg, train_loader)



