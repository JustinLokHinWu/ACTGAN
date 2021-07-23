import torch.nn as nn
import torch

class ACT_Discriminator(nn.Module):
    def __init__(self, args):
        super(ACT_Discriminator, self).__init__()

        self.gpu = args.cuda
        # Width of patch
        self.patch_width = args.disc_image_patch_size
        # Size of embedding used by transformer
        self.embed_dim = args.disc_embed_dim

        self.num_channels = args.num_channels

        # Number of patches in each direction
        self.num_patches_w = 8
        # Number of patches, i.e. square of number of patches in each direction
        self.num_patches = self.num_patches_w**2

        # First, we apply several convolutional layers. This is the same
        # as ACGAN
        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(args.num_channels, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, self.embed_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.embed_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )

        # Calculate and save 2d-positional encoding
        self.positional = self.calculate_positional_encoding_2d(8, self.embed_dim)
        if self.gpu:
            self.positional = self.positional.cuda()

        # Transformer layers
        self.trans = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=args.disc_heads,
                activation='gelu'),
            num_layers=args.disc_transformer_layers,
            norm=nn.LayerNorm(self.embed_dim)
        )
     
        # Transform into class and source outputs
        self.s = nn.Sequential(
            nn.Linear(self.embed_dim * self.num_patches, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.c = nn.Sequential(
            nn.Linear(self.embed_dim * self.num_patches, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, args.n_classes),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        # Make sure input is Batch x Channels x H x W
        x = x.view(-1,self.num_channels,32,32)

        # Apply convolutions
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)

        # Add positional encoding
        x = x + self.positional

        # Reshape for transformer (i.e. flatten to Batch x Num patches x Embedding dim)
        x = x.unfold(2,self.patch_width,self.patch_width
            ).unfold(3,self.patch_width,self.patch_width
            ).flatten(2,3
            ).transpose(1,2
            ).flatten(start_dim=2, end_dim=-1)
       
        # Pass through transformer encoder
        x = self.trans(x)

        # Pass results through FC layers to produce S and C
        s = self.s(x.flatten(start_dim=1))
        c = self.c(x.flatten(start_dim=1))

        return s, c
    
    def calculate_positional_encoding_2d(self, width, embed_dim):
        # Generate 2D positional encodings. Assume that the image is square
        pos = torch.arange(width)[..., None]
        dim = torch.arange(0, width//2, 2)[None, ...]

        term = pos / 10000**(torch.arange(0., embed_dim, 4) / embed_dim)

        encodings = torch.zeros((width, width, embed_dim))

        # Reuse these for x and y, just repeated in different axes
        sin_vals = torch.sin(term)
        cos_vals = torch.cos(term)

        quarter = embed_dim//4

        encodings[:,:,:quarter] = sin_vals.repeat(8,1,1)
        encodings[:,:,quarter:2*quarter] = cos_vals.repeat(8,1,1)
        encodings[:,:,2*quarter:3*quarter] = sin_vals.repeat(8,1,1).transpose(0,1)
        encodings[:,:,3*quarter:] = cos_vals.repeat(8,1,1).transpose(0,1)
        return encodings.permute(2,0,1)
