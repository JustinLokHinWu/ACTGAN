'''
Based on https://github.com/diegoalejogm/gans
'''
from torchvision import transforms, datasets

def get_cifar_data(cfg):
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    out_dir = '{}/cifar'.format(cfg.data_dir)
    train_data = datasets.CIFAR10(root=out_dir, train=True, transform=compose, download=True)
    return train_data

def get_mnist_data(cfg):
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(32),
         transforms.Normalize([0.5], [0.5])
        ])
    out_dir = '{}/mnist'.format(cfg.data_dir)
    train_data = datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)
    return train_data