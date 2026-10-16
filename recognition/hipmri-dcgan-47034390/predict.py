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
    week=config.get("week"),
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

fig, axes = plt.subplots(4, 8, figsize=(16, 9))

for i in range(16):
    row = i // 4
    col = i % 4

    # Real images: columns 0–3
    axes[row, col].imshow(
        real_images[i, 0].cpu().numpy(),
        cmap="gray",
        vmin=-1,
        vmax=1,
    )
    axes[row, col].axis("off")

    # Generated images: columns 4–7
    axes[row, col + 4].imshow(
        generated[i, 0].cpu().numpy(),
        cmap="gray",
        vmin=-1,
        vmax=1,
    )
    axes[row, col + 4].axis("off")

axes[0, 1].set_title("REAL", pad=12)
axes[0, 5].set_title("GENERATED", pad=12)


output_path = results_dir / "real_vs_generated.png"
output_path.parent.mkdir(parents=True, exist_ok=True)

plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches="tight")
# plt.show()

print(f"Comparison saved to: {output_path}")
print(f"Real image batch shape: {real_images.shape}")
print(f"Generated batch shape: {generated.shape}")