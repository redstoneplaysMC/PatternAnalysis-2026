import torch
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from modules import Generator

# Given an existing .pth file (trained Generator), generate a set of images via DCGAN.

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

with CONFIG_PATH.open("r") as f:
    all_config = yaml.safe_load(f)

config = {
    **all_config["defaults"],
    **all_config["dcgan"],
}

data_dir = Path(config["data_dir"])
results_dir = Path(config["results_dir"]) / "cifar10"
checkpoint_path = results_dir / "checkpoint.pth"
if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = config["latent_dim"]
checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

G = Generator(
    latent_dim=latent_dim,
    channels=3,
    image_size=32
).to(device)
G.load_state_dict(checkpoint["generator"])
G.eval()

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

    z = torch.randn(16, latent_dim, 1, 1, device=device)
    generated = G(z)

fig, axes = plt.subplots(4, 8, figsize=(12, 6))

for i in range(16):
    row = i // 4
    col = i % 4

    original = ((images[i].cpu().permute(1, 2, 0) + 1) / 2).clamp(0, 1)
    fake = ((generated[i].cpu().permute(1, 2, 0) + 1) / 2).clamp(0, 1)

    axes[row, col].imshow(original)
    axes[row, col].axis("off")

    axes[row, col + 4].imshow(fake)
    axes[row, col + 4].axis("off")

axes[0, 1].set_title("Original", pad=12)
axes[0, 5].set_title("Generated", pad=12)

plt.tight_layout()
plt.show()