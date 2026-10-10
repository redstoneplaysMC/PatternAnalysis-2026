# Implement the Baseline model for image generation, using some kind of VAE-encoder.
# Implement the actual DC-GAN model.

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import load_data_2D
import numpy as np

"""
Generator Class: transforms random noise vector into a full sized image through upsampling (Conv2d)
Parameters: latent_dim - dimension of random noise vector
            img_channels - number of output image channels in final layer (1 for grayscale)
            feature_maps - number of feature maps in each convolutional layer
"""
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()

        self.model = nn.Sequential(
            # (latent_dim, 4, 8)
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 4x8 -> 8x16
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 8x16 -> 16x32
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 16x32 -> 32x64
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 32x64 -> 64x128
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # 64x128 -> 128x256
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


"""Critic/Discriminator Class: evaluates the authenticity of an image (real or generated)
Parameters: img_channels - number of input image channels (1 for grayscale)
            feature_maps - number of feature maps in each convolutional layer
"""
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # 128x256 -> 64x128
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 64x128 -> 32x64
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x64 -> 16x32
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x32 -> 8x16
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x16 -> 4x8
            nn.Conv2d(512, 1, 4, 2, 1, bias=False),

            nn.Flatten(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)



"""Compute Gradient penalty for WGAN-GP
Parameters: critic - the discriminator/critic network
            real_images - batch of real images
            fake_images - batch of generated images
            device - torch device (cpu or cuda)
            lambda_gp - gradient penalty coefficient
"""

"""Init Weights for the network layers
Parameters: model - the neural network model whose weights need to be initialized
"""

"""Compute SSIM
Parameters: img1 - first image
            img2 - second image
"""
def compute_ssim(img1, img2, window_size=11, size_average=True):
    """
    Code was copied from samuel2003-coder.
    Compute Structured Similarity Index (SSIM) between two images
    
    Args:
        img1, img2: Images in range [-1, 1] or [0, 1]
        window_size: Size of the Gaussian window
        size_average: Whether to average the SSIM over the batch
    Returns:
        SSIM value
    """

    # Constants for stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor([
        torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
        for x in range(window_size)
    ])
    window = gauss / gauss.sum()
    window = window.unsqueeze(1)
    window = window.mm(window.t()).float().unsqueeze(0).unsqueeze(0)
    window = window.to(img1.device)
    
    # Ensure images are in [0, 1] range
    if img1.min() < 0:
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2
    
    # Calculate means
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=1)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=1)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=1) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=1) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)



if __name__ == "__main__":

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # G = Generator().to(device)
    # D = Discriminator().to(device)
    # print("Generator:", G)
    # print("Discriminator:", D)
    imageNames = [
        "test_input/case_040_week_0_slice_0.nii.gz",
        "test_input/case_004_week_0_slice_0.nii.gz"
    ]
    images = load_data_2D(imageNames, getAffines=False, normImage=True, categorical=False, dtype=np.float32, early_stop=False)
   
    # calculation of SSIM requires (batch, channels, height, width)
    # SSIM ranges roughly from:
    # 1.0 → essentially identical
    # 0.8–1.0 → very similar
    # 0.5–0.8 → moderately similar
    # 0.0–0.5 → increasingly different
    # 0.0633 → very little structural similarity

    images = torch.tensor(images).unsqueeze(1)  # Add channel dimension
    print(images.shape)
    img1 = images[0:1]
    img2 = images[1:2]

    img_min = min(img1.min(), img2.min())
    img_max = max(img1.max(), img2.max())

    img1 = 2 * (img1 - img_min) / (img_max - img_min) - 1
    img2 = 2 * (img2 - img_min) / (img_max - img_min) - 1

    images[0:1] = img1
    images[1:2] = img2

    print(torch.min(img1))
    print(torch.max(img1))
    print(compute_ssim(
        img1,
        img2,
        window_size=11,
        size_average=True,
    ))