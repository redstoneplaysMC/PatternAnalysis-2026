import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pathlib import Path

def save_generated_samples(
    decoder,
    epoch,
    latent_dim,
    device,
    results_dir,
    batch_size=16,
):
    decoder.eval()

    with torch.no_grad():
        z = torch.randn(batch_size, latent_dim, device=device)
        generated = decoder(z).cpu()

    generated = ((generated + 1) / 2).clamp(0, 1)

    rows = int(batch_size ** 0.5)
    cols = (batch_size + rows - 1) // rows

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 2, rows * 2),
        squeeze=False,
    )

    for i, ax in enumerate(axes.flat):
        ax.axis("off")

        if i < batch_size:
            ax.imshow(generated[i, 0], cmap="gray", vmin=0, vmax=1)

    fig.tight_layout(pad=0.1)
    fig.savefig(
        results_dir / f"epoch_{epoch:02d}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

def plot_training_losses(
    total_losses,
    reconstruction_losses,
    kl_losses,
    results_dir,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(total_losses, label="Total VAE Loss")
    ax.plot(reconstruction_losses, label="Reconstruction Loss")
    ax.plot(kl_losses, label="KL Divergence")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss per Image")
    ax.set_title("VAE Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        results_dir / "training_losses.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_validation_comparison(
    encoder,
    decoder,
    validate_dataloader,
    epoch,
    device,
    results_dir,
    count=8,
):
    encoder.eval()
    decoder.eval()

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    real_images = next(iter(validate_dataloader))
    if isinstance(real_images, (tuple, list)):
        real_images = real_images[0]

    real_images = real_images[:count].to(
        device=device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        mu, _ = encoder(real_images)
        reconstructed = decoder(mu)

    real_images = ((real_images.cpu() + 1) / 2).clamp(0, 1)
    reconstructed = ((reconstructed.cpu() + 1) / 2).clamp(0, 1)

    fig, axes = plt.subplots(2, len(real_images), figsize=(2 * len(real_images), 5))

    for i in range(len(real_images)):
        axes[0, i].imshow(real_images[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Original {i + 1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(reconstructed[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"Reconstruction {i + 1}")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Validation", fontsize=10)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=10)

    fig.suptitle(f"VAE Validation — Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(results_dir / f"validation_epoch_{epoch:02d}.png", dpi=150)
    plt.close(fig)
