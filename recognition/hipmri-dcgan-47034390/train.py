import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from pathlib import Path
from modules import Generator, Discriminator
import yaml
from utils import save_generated_samples, plot_training_losses
from dataset import HipMRIDataset

CONFIG_PATH = "config.yaml"

with open(CONFIG_PATH, "r") as f:
    all_config = yaml.safe_load(f)

config = {
    **all_config["defaults"],
    **all_config["hipmri_dcgan"],
}

data_dir = Path(config["data_dir"])
results_dir = Path(config["results_dir"])
results_dir.mkdir(parents=True, exist_ok=True)

# Dataset
print("Starting dataset initialisation...", flush=True)
dataset = HipMRIDataset(
    data_dir = data_dir,
    max_images = config.get("max_train_images"),
    norm_image = True,
    week = config.get("week"),
)

print("Dataset initialised. Loading first batch...", flush=True)
train_dataloader = DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=0, # Set to 0 while using the CPU.
    pin_memory=torch.cuda.is_available(),
)

images = next(iter(train_dataloader))
print("First batch loaded.", flush=True)
print("Number of 2D images:", len(dataset))
print("Batch shape:", images.shape)
print("Image range:", images.min().item(), images.max().item())

assert images.ndim == 4
assert images.shape[1:] == (1, 256, 128), (
    f"Expected [B, 1, 256, 128], got {images.shape}"
)

# Check device is CUDA if available
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Device:", device)

if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
    print(torch.cuda.get_device_name(0))

latent_dim = config["latent_dim"]
reference_images = torch.stack([
    dataset[i] for i in range(min(64, len(dataset)))
])

# save_image(
#     reference_images[::16],
#     results_dir / "reference_grid.png",
#     nrow=4,
#     normalize=True,
#     value_range=(-1, 1),
# )

# Models: These should support 128*256 Grayscale Images.
G = Generator(
    latent_dim=latent_dim,
    channels=1,
    image_size=(256, 128),
).to(device)

D = Discriminator(
    channels=1,
    image_size=(256, 128),
).to(device)

with torch.no_grad():
    fake = G(torch.randn(2, latent_dim, 1, 1, device=device))
    print("Generator output:", fake.shape)
    print("Discriminator output:", D(fake).shape)
    
optimizer_G = torch.optim.Adam(
    G.parameters(),
    lr=config["learning_rate_G"],
    betas=tuple(config["betas"]),
)

optimizer_D = torch.optim.Adam(
    D.parameters(),
    lr=config["learning_rate_D"],
    betas=tuple(config["betas"]),
)

# criterion = nn.BCEWithLogitsLoss()
criterion = nn.BCEWithLogitsLoss()

generator_losses = []
discriminator_losses = []

# Fixed noise for monitoring progress.
fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

if device.type == "cuda":
    torch.cuda.synchronize()

print("Training started!")
train_start = time.perf_counter()

for epoch in range(config["num_epochs"]):
    G.train()
    D.train()

    epoch_loss_G = 0.0
    epoch_loss_D = 0.0
    num_batches = 0

    for real_images in train_dataloader:
        if isinstance(real_images, (tuple, list)):
            real_images = real_images[0]

        real_images = real_images.to(device=device, dtype=torch.float32)
        batch_size = real_images.size(0)

        # Train discriminator
        optimizer_D.zero_grad(set_to_none=True)

        real_output = D(real_images).reshape(-1)
        loss_real = criterion(real_output, torch.ones_like(real_output))

        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_images = G(noise)

        fake_output = D(fake_images.detach()).reshape(-1)
        loss_fake = criterion(fake_output, torch.zeros_like(fake_output))

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # Train generator
        optimizer_G.zero_grad(set_to_none=True)

        fake_output = D(fake_images).reshape(-1)
        loss_G = criterion(fake_output, torch.ones_like(fake_output))

        loss_G.backward()
        optimizer_G.step()
        
        epoch_loss_G += loss_G.item()
        epoch_loss_D += loss_D.item()
        num_batches += 1

    # Compute mean losses for the epoch
    mean_G = epoch_loss_G / num_batches
    mean_D = epoch_loss_D / num_batches

    generator_losses.append(mean_G)
    discriminator_losses.append(mean_D)

    print(
        f"Epoch [{epoch + 1}/{config['num_epochs']}] "
        f"Loss D: {mean_D:.4f} "
        f"Loss G: {mean_G:.4f}"
    )

    # Save generated samples at selected epochs.
    if (epoch + 1) % 3 == 0 or epoch == 0:
        save_generated_samples(
            G,
            epoch + 1,
            latent_dim,
            device,
            results_dir,
            64,
        )

# Save checkpoint.
torch.save(
    {
        "generator": G.state_dict(),
        "discriminator": D.state_dict(),
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D": optimizer_D.state_dict(),
        "epoch": config["num_epochs"],
        "latent_dim": latent_dim,
    },
    results_dir / "checkpoint.pth",
)

# Generate final samples.
G.eval()

with torch.no_grad():
    z1 = torch.randn(8, 100, 1, 1)
    z2 = torch.randn(8, 100, 1, 1)

    images1 = G(z1)
    images2 = G(z2)

    print(images1.shape)
    print("Mean difference:", (images1 - images2).abs().mean().item())

with torch.no_grad():
    generated = G(fixed_noise).cpu()

print("Generated shape:", generated.shape)
print("Generated pixel std:", generated.std().item())
print(
    "Difference between first two images:",
    (generated[0] - generated[1]).abs().mean().item()
)
print(
    "Average variation across images:",
    generated.std(dim=0).mean().item()
)

# Convert [-1, 1] to [0, 1] for saving.
generated = ((generated + 1) / 2).clamp(0, 1)

# save_image(
#     generated[::16],
#     results_dir / "generated_grid.png",
#     nrow=4,
# )

# Plot generated images.
fig, axes = plt.subplots(4, 4, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    ax.imshow(generated[i, 0], cmap="gray", vmin=0, vmax=1)
    ax.axis("off")

plt.tight_layout()
plt.savefig(results_dir / "generated_samples.png", dpi=150)
plt.close(fig)

if device.type == "cuda":
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated(device)
    print(f"Peak GPU memory: {peak_memory / 1024**3:.2f} GB")

elapsed = time.perf_counter() - train_start
print(f"Training time: {elapsed:.2f} seconds")

plot_training_losses(
    generator_losses,
    discriminator_losses,
    results_dir,
)