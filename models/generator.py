import torch.nn as nn
import torch

class ACGAN_Generator(nn.Module):
    # Since the focus of our extension is to the discriminator, we did not
    # implement a generator from scratch.
    # This generator implementation is heavily based on an implementation
    # of ACGAN (https://arxiv.org/abs/1610.09585) by clvrai. 
    # Source link: https://github.com/clvrai/ACGAN-PyTorch/blob/master/network.py
    def __init__(self, args):
        super(ACGAN_Generator, self).__init__()
        input_size = args.n_classes + args.noise_size

        # First linear layer
        self.fc1 = nn.Linear(input_size, 384)
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, 4, 1, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, 2, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(48, args.num_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise, onehot):
        x = torch.cat((noise, onehot),dim=1)
        fc1 = self.fc1(x)
        fc1 = fc1.view(-1, 384, 1, 1)
        tconv2 = self.tconv2(fc1)
        tconv3 = self.tconv3(tconv2)
        tconv4 = self.tconv4(tconv3)
        tconv5 = self.tconv5(tconv4)
        output = tconv5

        return output
