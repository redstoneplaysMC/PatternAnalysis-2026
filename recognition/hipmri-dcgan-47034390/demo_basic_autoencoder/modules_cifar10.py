import torch.nn as nn
from torchvision.transforms.functional import resize
from torchmetrics.image.fid import FrechetInceptionDistance
import torch

# Encoder and decoder structure for the basic autoencoder. To run on the CIFAR10 Dataset.

class Encoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x.flatten(1))


class Decoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.features = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.features(self.fc(z))

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