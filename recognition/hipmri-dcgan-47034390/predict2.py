# Generate images from the .pth only; does not require access to the data directory.

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

from modules import Generator

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

with CONFIG_PATH.open("r") as f:
    all_config = yaml.safe_load(f)

config = {
    **all_config["defaults"],
    **all_config["hipmri_dcgan"],
}

results_dir = Path(config["results_dir"])
if not results_dir.is_absolute():
    results_dir = CONFIG_PATH.parent / results_dir

checkpoint_path = results_dir / "checkpoint.pth"

if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

results_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = config["latent_dim"]

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True,
)

G = Generator(latent_dim=latent_dim, channels=1).to(device)
G.load_state_dict(checkpoint["generator"])
G.eval()

count = 16

with torch.no_grad():
    z = torch.randn(count, latent_dim, 1, 1, device=device)
    generated = G(z).cpu()

# Generated-only grid
fig, axes = plt.subplots(4, 4, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    ax.imshow(
        generated[i, 0].numpy(),
        cmap="gray",
        vmin=-1,
        vmax=1,
    )
    ax.axis("off")

plt.tight_layout()

output_path = results_dir / "generated_diversity.png"
plt.savefig(output_path, dpi=150)
plt.close()

print(f"Device: {device}")
print(f"Generated shape: {generated.shape}")
print(f"Saved generated images to: {output_path}")
