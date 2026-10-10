import torch.nn as nn
from torchvision.transforms.functional import resize
from torchmetrics.image.fid import FrechetInceptionDistance
import torch

# https://medium.com/@masonthecount/using-a-dcgan-to-learn-the-mnist-dataset-in-order-to-generate-images-1ca616e45463
# AI-generated code.

class Generator(nn.Module):
    # Why? is this a DCGAN and not a basic GAN?
    def __init__(self, latent_dim=100):
        super().__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                latent_dim, 128,
                kernel_size=7,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                64, 1,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                1, 32,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                32, 64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                64, 1,
                kernel_size=7,
                stride=1,
                padding=0,
                bias=False
            )
        )

    def forward(self, x):
        return self.model(x).view(-1)

def compute_fid(reference_images, generated):
    def prepare_for_fid(images):

        # [N, 1, 28, 28]
        images = images.repeat(1, 3, 1, 1)

        # [N, 3, 299, 299]
        images = resize(images, [299, 299])

        # FID expects uint8 images in [0, 255]
        images = (images * 255).clamp(0, 255).to(torch.uint8)

        return images

    fid = FrechetInceptionDistance(
        feature=2048,
        normalize=False
    )
    batch_size = 8

    for i in range(0, len(reference_images), batch_size):

        real_batch = reference_images[i:i + batch_size]

        real_batch = prepare_for_fid(real_batch)

        fid.update(
            real_batch,
            real=True
        )


    for i in range(0, len(generated), batch_size):

        fake_batch = generated[i:i + batch_size]

        fake_batch = prepare_for_fid(fake_batch)

        fid.update(
            fake_batch,
            real=False
        )


    fid_score = fid.compute()

    print(f"FID: {fid_score.item():.4f}")