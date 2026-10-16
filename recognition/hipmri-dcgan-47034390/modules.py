# Ts vibe-coded lol. Fix it and make it produce something passable before submitting it to rangpur.
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Generator Class: transforms random noise vector into a full sized image through upsampling (Conv2d)
Parameters: latent_dim - dimension of random noise vector
            img_channels - number of output image channels in final layer (1 for grayscale)
            feature_maps - number of feature maps in each convolutional layer
"""
class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=1, image_size=(256, 128)):
        super().__init__()

        if image_size != (256, 128):
            raise ValueError("This generator outputs 256x128 images")

        self.model = nn.Sequential(
            # 1x1 -> 8x4
            nn.ConvTranspose2d(
                latent_dim, 512, kernel_size=(8, 4),
                stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 8x4 -> 16x8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 16x8 -> 32x16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 32x16 -> 64x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64x32 -> 128x64
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 128x64 -> 256x128
            nn.ConvTranspose2d(32, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)


"""Critic/Discriminator Class: evaluates the authenticity of an image (real or generated)
Parameters: img_channels - number of input image channels (1 for grayscale)
            feature_maps - number of feature maps in each convolutional layer
"""
class Discriminator(nn.Module):
    def __init__(self, channels=1, image_size=(256, 128)):
        super().__init__()

        if image_size != (256, 128):
            raise ValueError("This discriminator expects 256x128 images")

        self.features = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=(16, 8), stride=1, padding=0, bias=False),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.features(x)
