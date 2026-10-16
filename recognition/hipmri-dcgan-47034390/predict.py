import torch
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

from torch.utils.data import DataLoader, Subset
from modules import Generator
from dataset import HipMRIDataset

import random

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

with CONFIG_PATH.open("r") as f:
    all_config = yaml.safe_load(f)

config = {
    **all_config["defaults"],
    **all_config["hipmri_dcgan"],
}

data_dir = Path(config["data_dir"])
if not data_dir.is_absolute():
    data_dir = CONFIG_PATH.parent / data_dir

results_dir = Path(config["results_dir"])
if not results_dir.is_absolute():
    results_dir = CONFIG_PATH.parent / results_dir

# Match the checkpoint path used by your training script.
checkpoint_path = results_dir / "checkpoint.pth"

if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path}"
    )

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
latent_dim = config["latent_dim"]

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False,
)

G = Generator(latent_dim=latent_dim, channels=1).to(device)
G.load_state_dict(checkpoint["generator"])
G.eval()

dataset = HipMRIDataset(
    data_dir=data_dir,
    max_images=config.get("max_train_images"),
    norm_image=config.get("norm_image", True),
    week=4,
)

count = min(16, len(dataset))
indices = random.sample(range(len(dataset)), count)

dataloader = DataLoader(
    Subset(dataset, indices),
    batch_size=count,
    shuffle=False,
)

real_images = next(iter(dataloader)).to(device)

# Generate the same number of images as the reference batch.
with torch.no_grad():
    z = torch.randn(count, latent_dim, 1, 1, device=device)
    generated = G(z)


# Plotting

# Real vs generated comparison
fig, axes = plt.subplots(4, 8, figsize=(16, 9))

for i in range(16):
    row, col = i // 4, i % 4

    axes[row, col].imshow(
        real_images[i, 0].cpu().numpy(),
        cmap="gray", vmin=-1, vmax=1,
    )
    axes[row, col].axis("off")

    axes[row, col + 4].imshow(
        generated[i, 0].cpu().numpy(),
        cmap="gray", vmin=-1, vmax=1,
    )
    axes[row, col + 4].axis("off")

axes[0, 1].set_title("REAL")
axes[0, 5].set_title("GENERATED")

plt.tight_layout()
plt.savefig(results_dir / "real_vs_generated.png", dpi=150)
plt.close()

# Generated-only grid to inspect diversity
fig, axes = plt.subplots(4, 4, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    ax.imshow(
        generated[i, 0].cpu().numpy(),
        cmap="gray", vmin=-1, vmax=1,
    )
    ax.axis("off")

plt.tight_layout()
plt.savefig(results_dir / "generated_diversity.png", dpi=150)
plt.show()

print(f"Saved comparison to: {results_dir / 'real_vs_generated.png'}")
print(f"Saved diversity grid to: {results_dir / 'generated_diversity.png'}")