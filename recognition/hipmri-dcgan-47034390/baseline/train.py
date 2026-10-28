import time
import torch
import matplotlib.pyplot as plt
import yaml
from torch.utils.data import DataLoader
from pathlib import Path
from baseline.modules import Encoder, Decoder
from baseline.utils import save_generated_samples, plot_training_losses, save_validation_comparison
from dataset import HipMRIDataset
from torchvision.utils import save_image
import torch
import torch.nn.functional as F

# Run from the project root.

IMAGE_SIZE = (256, 128)
IMAGE_CHANNELS = 1
SAMPLE_INTERVAL = 3
CONFIG_PATH = Path("config.yaml")

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    profile = "hipmri_vae" if raw["defaults"]["rangpur"] else "hipmri_vae_cpu"
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
        data_dir = Path(config["data_dir"]),
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
        data_dir = Path(config["data_dir"]),
        max_images = None,
        norm_image = True,
        week = config["week"]
    )

    print("Dataset initialised. Loading first batch...", flush=True)
    validation_dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0, # Set to 0 while using the CPU.
        pin_memory=torch.cuda.is_available(),
    )

    images = next(iter(validation_dataloader))
    print("First batch loaded.", flush=True)
    print("Number of 2D images:", len(dataset))
    print("Batch shape:", images.shape)
    print("Image range:", images.min().item(), images.max().item())
    assert images.ndim == 4
    assert images.shape[1:] == (1, 256, 128), (
        f"Expected [B, 1, 256, 128], got {images.shape}"
    )

    return validation_dataloader

def initialize_training_model(config, latent_dim, device):
    encoder = Encoder(
        latent_dim=latent_dim,
        channels=1,
        image_size=(256, 128),
    ).to(device)

    decoder = Decoder(
        latent_dim=latent_dim,
        channels=1,
        image_size=(256, 128),
    ).to(device)

    with torch.no_grad():
        z = torch.randn(2, latent_dim, device=device)
        fake = decoder(z)
        mu, logvar = encoder(fake)

        print("Decoder output:", fake.shape)
        print("Encoder mean:", mu.shape)
        print("Encoder logvar:", logvar.shape)


    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config["learning_rate"],
        betas=tuple(config["betas"]),
    )

    return encoder, decoder, optimizer


def train_one_epoch(
    train_dataloader,
    encoder,
    decoder,
    optimizer,
    device,
    beta=1.0,
):
    encoder.train()
    decoder.train()

    total_loss = 0.0
    total_reconstruction = 0.0
    total_kl = 0.0
    num_batches = 0

    for real_images in train_dataloader:
        if isinstance(real_images, (tuple, list)):
            real_images = real_images[0]

        real_images = real_images.to(
            device=device,
            dtype=torch.float32,
        )

        optimizer.zero_grad(set_to_none=True)

        # Encode into a latent distribution.
        mu, logvar = encoder(real_images)

        # Reparameterisation trick.
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std

        # Reconstruct the input image.
        reconstructed = decoder(z)

        # Reconstruction loss.
        reconstruction_loss = F.mse_loss(
            reconstructed,
            real_images,
            reduction="sum",
        ) / real_images.size(0)

        # KL divergence against N(0, I).
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        ) / real_images.size(0)

        # Combined VAE objective.
        loss = reconstruction_loss + beta * kl_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_reconstruction += reconstruction_loss.item()
        total_kl += kl_loss.item()
        num_batches += 1

    if num_batches == 0:
        raise ValueError("Training dataloader contains no batches.")

    return (
        total_loss / num_batches,
        total_reconstruction / num_batches,
        total_kl / num_batches,
    )

def validate_one_epoch(
    validate_dataloader,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    device,
    beta=1.0,
):
    encoder.eval()
    decoder.eval()

    total_loss = 0.0
    total_reconstruction = 0.0
    total_kl = 0.0
    num_batches = 0

    with torch.no_grad():
        for real_images in validate_dataloader:
            if isinstance(real_images, (tuple, list)):
                real_images = real_images[0]

            real_images = real_images.to(
                device=device,
                dtype=torch.float32,
            )

            mu, logvar = encoder(real_images)

            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std

            reconstructed = decoder(z)

            reconstruction_loss = F.mse_loss(
                reconstructed,
                real_images,
                reduction="sum",
            ) / real_images.size(0)

            kl_loss = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp()
            ) / real_images.size(0)

            loss = reconstruction_loss + beta * kl_loss

            total_loss += loss.item()
            total_reconstruction += reconstruction_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1

    if num_batches == 0:
        raise ValueError("Validation dataloader contains no batches.")

    return (
        total_loss / num_batches,
        total_reconstruction / num_batches,
        total_kl / num_batches,
    )
    

def save_checkpoint(
    encoder,
    decoder,
    optimizer,
    epoch,
    latent_dim,
    results_dir,
):
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "latent_dim": latent_dim,
        },
        results_dir / "checkpoint.pth",
    )

def train_loop(
    config,
):
    # Setup.
    latent_dim = config["latent_dim"]
    train_dataloader = initialize_dataloader(config)
    device = configure_device()
    if config["validate"]:
        validate_dataloader = initialize_validation_dataloader(config)

    encoder, decoder, optimizer = (
        initialize_training_model(config, latent_dim, device)
    )

    num_epochs = config["num_epochs"]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    print("Training started!")
    train_start = time.perf_counter()

    # Main training loop.
    total_losses = []
    reconstruction_losses = []
    kl_losses = []

    for epoch in range(1, num_epochs + 1):
        mean_loss, mean_reconstruction, mean_kl = train_one_epoch(
            train_dataloader,
            encoder,
            decoder,
            optimizer,
            device,
            beta=config.get("beta", 1.0),
        )

        total_losses.append(mean_loss)
        reconstruction_losses.append(mean_reconstruction)
        kl_losses.append(mean_kl)

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Loss: {mean_loss:.4f} "
            f"Reconstruction: {mean_reconstruction:.4f} "
            f"KL: {mean_kl:.4f}"
        )
        # Save generated samples for every [sample_interval] epochs.
        if epoch % config["sample_interval"] == 0 or epoch == 1:
            save_generated_samples(
                decoder,
                epoch,
                latent_dim,
                device,
                Path(config["results_dir"]),
                64,
            )

        if config["validate"] and epoch % config["validate_every"] == 0:
            val_loss, val_reconstruction, val_kl = validate_one_epoch(
                validate_dataloader,  # type: ignore
                encoder,
                decoder,
                device,
            )

            save_validation_comparison(
                encoder,
                decoder,
                validate_dataloader,  # type: ignore
                epoch,
                device,
                Path(config["results_dir"]),
                count=8,
            )

            print(
                f"Validation - Epoch [{epoch}/{num_epochs}] "
                f"Loss: {val_loss:.4f} "
                f"Reconstruction: {val_reconstruction:.4f} "
                f"KL: {val_kl:.4f}"
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
        encoder,
        decoder,
        optimizer,
        num_epochs,
        latent_dim,
        Path(config["results_dir"]),
    )

    return encoder, decoder, total_losses, reconstruction_losses, kl_losses

def create_plot(decoder, results_dir, total_losses, reconstruction_losses, kl_losses, latent_dim, device):
    # Generate final samples.
    decoder.eval()

    with torch.no_grad():
        noise = torch.randn(64, latent_dim, device=device)
        generated = decoder(noise).cpu()

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
        total_losses,
        reconstruction_losses,
        kl_losses,
        results_dir,
    )
        
def main():
    config = load_config()
    set_seed(config.get("seed"))
    results_dir = Path(config["results_dir"])
    device = configure_device()
    results_dir.mkdir(parents=True, exist_ok=True)
    encoder, decoder, total_losses, reconstruction_losses, kl_losses = \
        train_loop(config)
    create_plot(
        decoder, 
        results_dir, 
        total_losses, 
        reconstruction_losses, 
        kl_losses, 
        config["latent_dim"],
        device
    )
    return encoder, decoder, total_losses, reconstruction_losses, kl_losses

if __name__ == "__main__":
    main()