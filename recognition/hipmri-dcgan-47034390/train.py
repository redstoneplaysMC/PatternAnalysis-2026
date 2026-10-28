import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
from modules import Generator, Discriminator
import yaml
from utils import save_generated_samples, plot_training_losses, save_validation_comparison
from dataset import HipMRIDataset
from torchvision.utils import save_image

IMAGE_SIZE = (256, 128)
IMAGE_CHANNELS = 1
SAMPLE_COUNT = 64
SAMPLE_INTERVAL = 3
CONFIG_PATH = Path("config.yaml")


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    profile = "hipmri_dcgan" if raw["defaults"]["rangpur"] else "hipmri_dcgan_cpu"
    return {**raw["defaults"], **raw[profile]}

def configure_device():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        print(torch.cuda.get_device_name(0))

    return device

def initialize_dataloader(config):
    print("Starting dataset initialisation...", flush=True)
    dataset = HipMRIDataset(
        data_dir = config["data_dir"],
        max_images = config.get("max_train_images"),
        norm_image = True,
        week = config["week"]
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

    return train_dataloader

def initialize_validation_dataloader(config):
    print("Starting dataset initialisation...", flush=True)
    dataset = HipMRIDataset(
        data_dir = Path(config["validate_dir"]),
        max_images = None,
        norm_image = True,
        week = config["week"]
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

    return train_dataloader


def initialize_training_model(config, latent_dim, device):
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

    return G, D, optimizer_G, optimizer_D

def train_one_epoch(
    train_dataloader,
    G: torch.nn.Module,
    D: torch.nn.Module,
    optimizer_G,
    optimizer_D,
    criterion,
    latent_dim,
    device,
):

    # Set the models to training mode, and initialize epoch losses and batch counter.
    G.train()
    D.train()
    epoch_loss_G = 0.0
    epoch_loss_D = 0.0
    num_batches = 0

    for real_images in train_dataloader:
        # Ensure real_images is a tensor of shape [B, 1, 256, 128] before moving to device.
        if isinstance(real_images, (tuple, list)):
            real_images = real_images[0]
            
        # Move the images to the device. 
        real_images = real_images.to(
            device=device,
            dtype=torch.float32,
        )
        batch_size = real_images.size(0)

        # Train discriminator on real and fake images. 
        # Zero gradients first, so that previous updates do not affect the current step.
        optimizer_D.zero_grad(set_to_none=True)
        real_output = D(real_images).reshape(-1)
        loss_real = criterion(
            real_output,
            torch.ones_like(real_output),
        )

        noise = torch.randn(
            batch_size, latent_dim, 1, 1, device=device
        )
        fake_images = G(noise)

        fake_output = D(fake_images.detach()).reshape(-1)
        loss_fake = criterion(
            fake_output,
            torch.zeros_like(fake_output),
        )

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # Train generator.
        optimizer_G.zero_grad(set_to_none=True)

        fake_output = D(fake_images).reshape(-1)
        loss_G = criterion(
            fake_output,
            torch.ones_like(fake_output),
        )

        loss_G.backward()
        optimizer_G.step()

        epoch_loss_G += loss_G.item()
        epoch_loss_D += loss_D.item()
        num_batches += 1

    if num_batches == 0:
        raise ValueError("Training dataloader contains no batches.")

    return (
        epoch_loss_G / num_batches,
        epoch_loss_D / num_batches,
    )


def validate_one_epoch(
    validate_dataloader,
    G: torch.nn.Module,
    D: torch.nn.Module,
    criterion,
    latent_dim,
    device,
):
    G.eval()
    D.eval()

    epoch_loss_G = 0.0
    epoch_loss_D = 0.0
    num_batches = 0

    with torch.no_grad():
        for real_images in validate_dataloader:
            if isinstance(real_images, (tuple, list)):
                real_images = real_images[0]

            real_images = real_images.to(
                device=device,
                dtype=torch.float32,
            )
            batch_size = real_images.size(0)

            real_output = D(real_images).reshape(-1)
            loss_real = criterion(
                real_output,
                torch.ones_like(real_output),
            )

            noise = torch.randn(
                batch_size, latent_dim, 1, 1, device=device
            )
            fake_images = G(noise)

            fake_output = D(fake_images).reshape(-1)
            loss_fake = criterion(
                fake_output,
                torch.zeros_like(fake_output),
            )

            loss_D = loss_real + loss_fake

            fake_output_for_G = D(fake_images).reshape(-1)
            loss_G = criterion(
                fake_output_for_G,
                torch.ones_like(fake_output_for_G),
            )

            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()
            num_batches += 1

    if num_batches == 0:
        raise ValueError("Validation dataloader contains no batches.")

    return (
        epoch_loss_G / num_batches,
        epoch_loss_D / num_batches,
    )
    

def save_checkpoint(
    G,
    D,
    optimizer_G,
    optimizer_D,
    epoch,
    latent_dim,
    results_dir,
):
    torch.save(
        {
            "generator": G.state_dict(),
            "discriminator": D.state_dict(),
            "optimizer_G": optimizer_G.state_dict(),
            "optimizer_D": optimizer_D.state_dict(),
            "epoch": epoch,
            "latent_dim": latent_dim,
        },
        results_dir / "checkpoint.pth",
    )

def train_loop(config):
    # Setup.
    latent_dim = config["latent_dim"]
    train_dataloader = initialize_dataloader(config)
    device = configure_device()

    if config["validate"]:
        validate_dataloader = initialize_validation_dataloader(config)

    G, D, optimizer_G, optimizer_D = (
        initialize_training_model(config, latent_dim, device)
    )

    criterion = nn.BCEWithLogitsLoss()
    num_epochs = config["num_epochs"]

    generator_losses = []
    discriminator_losses = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    print("Training started!")
    train_start = time.perf_counter()

    # Main training loop.
    for epoch in range(1, num_epochs + 1):
        mean_G, mean_D = train_one_epoch(
            train_dataloader,
            G,
            D,
            optimizer_G,
            optimizer_D,
            criterion,
            latent_dim,
            device,
        )

        # Append the mean losses for this epoch to the respective lists, then log the epoch statistics.
        generator_losses.append(mean_G)
        discriminator_losses.append(mean_D)
        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Loss D: {mean_D:.4f} "
            f"Loss G: {mean_G:.4f}"
        )

        # Save generated samples for every [sample_interval] epochs.
        if epoch % config["sample_interval"] == 0 or epoch == 1:
            save_generated_samples(
                G,
                epoch,
                latent_dim,
                device,
                Path(config["results_dir"]),
                64,
            )
            
        if config["validate"] and epoch % config["validate_every"] == 0:
            val_G, val_D = validate_one_epoch(
                validate_dataloader, # type:ignore
                G,
                D,
                criterion,
                latent_dim,
                device,
            )

            save_validation_comparison(
                    G,
                    validate_dataloader, #type:ignore
                    epoch,
                    latent_dim,
                    device,
                    Path(config["results_dir"]),
                    count=8,
                )

            print(
                f"Validation - Epoch [{epoch}/{num_epochs}] "
                f"Loss D: {val_D:.4f} "
                f"Loss G: {val_G:.4f}"
            )

    # Final statistics.
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated(device)
        print(f"Peak GPU memory: {peak_memory / 1024**3:.2f} GB")

    elapsed = time.perf_counter() - train_start
    print(f"Training time: {elapsed:.2f} seconds")

    # Save final checkpoint as a pth file.
    save_checkpoint(
        G,
        D,
        optimizer_G,
        optimizer_D,
        num_epochs,
        latent_dim,
        Path(config["results_dir"]),
    )

    return G, D, generator_losses, discriminator_losses

def create_plot(G, results_dir, generator_losses, discriminator_losses, latent_dim):
    # Generate final samples.
    G.eval()

    with torch.no_grad():
        noise = torch.randn(64, latent_dim, 1, 1)
        generated = G(noise).cpu()
        
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

    save_image(
        generated,
        results_dir / "generated_grid.png",
        nrow=4,
    )

    # Plot generated images.
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        ax.imshow(generated[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(results_dir / "generated_samples.png", dpi=150)
    plt.close(fig)

    plot_training_losses(
        generator_losses,
        discriminator_losses,
        results_dir,
    )
        
def main():
    config = load_config()
    set_seed(config.get("seed"))
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    G, D, generator_losses, discriminator_losses = train_loop(config)
    create_plot(
        G, 
        results_dir, 
        generator_losses, 
        discriminator_losses, 
        config["latent_dim"])
    return G, D, generator_losses, discriminator_losses

if __name__ == "__main__":
    main()