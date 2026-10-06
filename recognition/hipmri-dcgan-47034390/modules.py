# Implement the Baseline model for image generation, using some kind of VAE-encoder.
# Implement the actual DC-GAN model.

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Generator Class: transforms random noise vector into a full sized image through upsampling (Conv2d)
Parameters: latent_dim - dimension of random noise vector
            img_channels - number of output image channels in final layer (1 for grayscale)
            feature_maps - number of feature maps in each convolutional layer
"""

"""Critic/Discriminator Class: evaluates the authenticity of an image (real or generated)
Parameters: img_channels - number of input image channels (1 for grayscale)
            feature_maps - number of feature maps in each convolutional layer
"""

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

