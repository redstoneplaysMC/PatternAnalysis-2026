import torch
import matplotlib.pyplot as plt
from pathlib import Path

import yaml

from train_cifar10 import Generator, latent_dim, device

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

results_dir = Path(config["results_dir"]) / "cifar10"
    
checkpoint = torch.load(
    results_dir / "checkpoint.pth",
    map_location=device
)

G = Generator(
    latent_dim=latent_dim,
    channels=3,
    image_size=32
).to(device)
G.load_state_dict(checkpoint["generator"])
G.eval()

with torch.no_grad():
    z = torch.randn(16, latent_dim, 1, 1, device=device)
    generated = G(z)
    
fig, axes = plt.subplots(4, 4, figsize=(6, 6))

# for i, ax in enumerate(axes.flat):
#     ax.imshow(generated[i, 0].cpu(), cmap="gray")
#     ax.axis("off")

for i, ax in enumerate(axes.flat):
    img = generated[i].cpu().permute(1, 2, 0)
    img = (img.clamp(-1, 1) + 1) / 2
    ax.imshow(img)
    ax.axis("off")

plt.tight_layout()
plt.show()