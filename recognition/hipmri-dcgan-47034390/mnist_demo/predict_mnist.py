import torch
import matplotlib.pyplot as plt
from pathlib import Path

from train_mnist import Generator, latent_dim, device

results_dir = Path("results/mnist")

checkpoint = torch.load(
    results_dir / "checkpoint.pth",
    map_location=device
)

G = Generator(latent_dim).to(device)
G.load_state_dict(checkpoint["generator"])
G.eval()

with torch.no_grad():
    z = torch.randn(16, latent_dim, 1, 1, device=device)
    generated = G(z)
    
fig, axes = plt.subplots(4, 4, figsize=(6, 6))

for i, ax in enumerate(axes.flat):
    ax.imshow(generated[i, 0].cpu(), cmap="gray")
    ax.axis("off")

plt.tight_layout()
plt.show()