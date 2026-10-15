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
            nn.ConvTranspose2d(latent_dim, 512, (4, 2), 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            # nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, channels, 4, 2, 1, bias=False),
            nn.Tanh()
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
        return ssim_map

