from typing_extensions import get_args
import torch
from attrdict import AttrDict
import dataloader.datasets
from models.discriminator import ACT_Discriminator
from models.generator import ACGAN_Generator

from training_loop import train

if __name__=='__main__':
    # Load config
    cfg_dict = {
        'noise_size': 2,
        'n_classes': 10, 
        'image_size': 32,  
        'num_channels': 3, # Greyscale=1, RGB=3
        'batch_size': 100,
        'cuda': True,
        'dataset': 'cifar',
        'data_dir': './datasets/cifar10/',
         "generator": ACGAN_Generator,
        "discriminator": ACT_Discriminator,
        "noise_size": 100,
        "beta1": 0.9,
        "beta2": 0.99,
        "momentum": 0.9,
        "num_epochs": 1,
        "d_lr": 0.0002,
        "g_lr": 0.0002,
        "disc_image_patch_size": 1,  # for transformer model
        "disc_embed_dim": 128,
        "disc_heads": 16,
        "disc_transformer_layers": 6,
        "logger_data_name": 'CIFAR10'
    }
    cfg = AttrDict(cfg_dict)

    # Load dataset
    if cfg.dataset=='cifar':
        train_data, test_data = dataloader.datasets.get_cifar_data(cfg)
    elif cfg.dataset=='mnist':
        train_data, test_data = dataloader.datasets.get_mnist_data(cfg)
    
    # Train Loader
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True)

    # Test Loader
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=cfg.batch_size,
        shuffle=True)

    # Run training loop
    train(cfg, train_loader)



