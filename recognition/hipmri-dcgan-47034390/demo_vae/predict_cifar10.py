import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import yaml

from demo_vae.modules_cifar10 import Encoder, Decoder, reparameterize

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

with CONFIG_PATH.open("r") as f:
    all_config = yaml.safe_load(f)

config = {
    **all_config["defaults"],
    **all_config["vae"],
}

data_dir = Path(config["data_dir"])
results_dir = Path(config["results_dir"]) / "cifar10"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = results_dir / "checkpoint.pth"
if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location=device)

encoder = Encoder(config["latent_dim"]).to(device)
decoder = Decoder(config["latent_dim"]).to(device)

encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])

encoder.eval()
decoder.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = Subset(
    datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    ),
    range(16),
)

dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

with torch.no_grad():
    images, _ = next(iter(dataloader))
    images = images.to(device)
    mu, logvar = encoder(images)
    z = reparameterize(mu, logvar)
    reconstructed = decoder(z)

fig, axes = plt.subplots(4, 8, figsize=(12, 6))

for i in range(16):
    original = ((images[i].cpu().permute(1, 2, 0) + 1) / 2).clamp(0, 1)
    reconstruction = (
        (reconstructed[i].cpu().permute(1, 2, 0) + 1) / 2
    ).clamp(0, 1)

    axes[i // 4, (i % 4) * 2].imshow(original)
    axes[i // 4, (i % 4) * 2].set_title("Original")
    axes[i // 4, (i % 4) * 2].axis("off")

    axes[i // 4, (i % 4) * 2 + 1].imshow(reconstruction)
    axes[i // 4, (i % 4) * 2 + 1].set_title("Reconstruction")
    axes[i // 4, (i % 4) * 2 + 1].axis("off")

plt.tight_layout()
plt.savefig(results_dir / "reconstruction.png") 
plt.show()

# Generate new images from random latent vectors. Will be low quality.

latent_dim = 256

encoder = Encoder(latent_dim=latent_dim).to(device)
decoder = Decoder(latent_dim=latent_dim).to(device)

encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])

num_images = 16

with torch.no_grad():
    z = torch.randn(num_images, latent_dim, device=device)
    generated = decoder(z)

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle("Generated Images, VAE")

for i, ax in enumerate(axes.flat):
    image = ((generated[i].cpu().permute(1, 2, 0) + 1) / 2).clamp(0, 1)
    ax.imshow(image)
    ax.axis("off")

plt.tight_layout()
plt.savefig(results_dir / "generated.png")
plt.show()