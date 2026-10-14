import torch.nn as nn
from torchvision.transforms.functional import resize
from torchmetrics.image.fid import FrechetInceptionDistance
import torch

# https://medium.com/@masonthecount/using-a-dcgan-to-learn-the-mnist-dataset-in-order-to-generate-images-1ca616e45463
# AI-generated code.

class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=3, image_size=32):
        super().__init__()
        if image_size != 32:
            raise ValueError("This architecture outputs 32x32 images")

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 8, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, channels=3, image_size=32):
        super().__init__()
        if image_size != 32:
            raise ValueError("This architecture expects 32x32 images")

        self.model = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 1, 8, 1, 0, bias=False)
        )

    def forward(self, x):
        return self.model(x).view(-1)

def compute_fid(reference_images, generated):
    
    def prepare_for_fid(images):
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        images = resize(images, [299, 299])

        images = ((images + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)

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