import random
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import yaml

from baseline.modules import Decoder

# Prediction script for generating and visualizing images using a trained DCGAN generator.
# A different predict must be used for the baseline.

# Path to the configuration file and other constants for prediction
CONFIG_PATH = Path("config.yaml")
INCLUDE_REFERENCE_DATASET = True
COUNT = 16

def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    profile = "hipmri_dcgan" if raw["defaults"]["rangpur"] else "hipmri_dcgan_cpu"
    return {**raw["defaults"], **raw[profile]}

def load_checkpoint(checkpoint_path, device, config):
    """Load the generator model from a checkpoint.

    Args:
        checkpoint_path (Path): Path to the checkpoint file.
        device (torch.device): Device to load the model onto.
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        Generator: The loaded generator model.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    G = Decoder(latent_dim=config["latent_dim"], channels=1).to(device)
    G.load_state_dict(checkpoint["decoder"])
    G.eval()
    return G

def resolve_path(value):
    """Resolve a file path relative to the configuration file.

    Args:
        value (str or Path): The file path to resolve.

    Returns:
        Path: The resolved absolute file path.
    """
    path = Path(value)
    return path if path.is_absolute() else CONFIG_PATH.parent / path

def load_real_images(data_dir, config):
    """Load a subset of real images from the dataset.

    Args:
        data_dir (Path): Path to the directory containing the dataset.
        config (dict): Configuration dictionary containing dataset parameters.

    Returns:
        torch.Tensor: A batch of real images.
    """
    from dataset import HipMRIDataset
    from torch.utils.data import DataLoader, Subset

    dataset = HipMRIDataset(
        data_dir=data_dir,
        max_images=config.get("max_train_images"),
        norm_image=config.get("norm_image", True),
        week=config.get("week"),
    )
    indices = random.sample(range(len(dataset)), COUNT)
    return next(iter(DataLoader(Subset(dataset, indices), batch_size=COUNT)))


def main():
    config = load_config()

    data_dir = resolve_path(config["data_dir"])
    results_dir = resolve_path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = config["latent_dim"]
    checkpoint_path = results_dir / "checkpoint.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    G = load_checkpoint(checkpoint_path, device, config)
    with torch.no_grad():
        z = torch.randn(COUNT, latent_dim, 1, 1, device=device)
        generated = G(z).cpu()

    if INCLUDE_REFERENCE_DATASET:
        real_images = load_real_images(data_dir, config)
        fig, axes = plt.subplots(4, 8, figsize=(16, 9))

        for i in range(COUNT):
            for col, images in ((i % 4, real_images), (i % 4 + 4, generated)):
                ax = axes[i // 4, col]
                ax.imshow(images[i, 0].numpy(), cmap="gray", vmin=-1, vmax=1)
                ax.axis("off")

        axes[0, 1].set_title("REAL")
        axes[0, 5].set_title("GENERATED")
    else:
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))

        for ax, image in zip(axes.flat, generated):
            ax.imshow(image[0].numpy(), cmap="gray", vmin=-1, vmax=1)
            ax.axis("off")

    plt.tight_layout()
    output_path = results_dir / "generated_diversity.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Generated shape: {generated.shape}")
    print(f"Device: {device}")
    print(f"Saved images to: {output_path}")
    # Implement metrics.py later.

if __name__ == "__main__":
    main()