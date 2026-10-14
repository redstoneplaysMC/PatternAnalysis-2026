import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from modules_cifar10 import Encoder, Decoder, compute_fid
import yaml
from utils_cifar10 import plot_training_losses, save_and_plot_generated

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

with CONFIG_PATH.open("r") as f:
    all_config = yaml.safe_load(f)

config = {
    **all_config["defaults"],
    **all_config["basic_autoencoder"],
}

data_dir = Path(config["data_dir"])
results_dir = Path(config["results_dir"]) / "cifar10"
results_dir.mkdir(parents=True, exist_ok=True)
num_to_save = max(64, config["num_generated"])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = Subset(
    datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    ),
    range(config["size_train_dataset"])
)

reference_dataset = Subset(
    datasets.CIFAR10(
        # Create the reference dataset for evaluation purposes
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    ),
    range(config["size_reference_dataset"])
)

train_dataloader = DataLoader(
    # Create a DataLoader for the training dataset with the specified batch size and shuffling
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True
)

reference_dataloader = DataLoader(
    reference_dataset,
    batch_size=config["batch_size"] ,
    shuffle=False
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Device:", device)

if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
    print(torch.cuda.get_device_name(0))
    

latent_dim = config["latent_dim"]
images, labels = next(iter(train_dataloader))
print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Images min: {images.min()}")
print(f"Images max: {images.max()}")

if __name__ == "__main__":

    channels=3
    image_size=32
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config["learning_rate"],
        betas=tuple(config["betas"])
    )
    
    reconstruction_losses = []
    criterion = nn.MSELoss()
    if device.type == "cuda":
        torch.cuda.synchronize()
        print(f"Training Timer started!")
    train_start = time.perf_counter()

    for epoch in range(config["num_epochs"]):
        encoder.train()
        decoder.train()
        epoch_loss = 0.0
        epoch_start = time.perf_counter()
        for real_images, _ in train_dataloader:
            real_images = real_images.to(device)
            optimizer.zero_grad()

            latent = encoder(real_images)
            reconstructed = decoder(latent)

            loss = criterion(reconstructed, real_images)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_dataloader)
        reconstruction_losses.append(epoch_loss)

        print(
            f"Reconstruction Loss: {epoch_loss:.4f}, Epoch time: {time.perf_counter() - epoch_start:.2f} seconds"
        )
        epoch_end = time.perf_counter()
        print(f"Epoch [{epoch + 1}/{config['num_epochs']}] completed.")

        # save_and_plot_generated(
        #     reconstructed[:num_to_save], # type:ignore
        #     images[:num_to_save],
        #     results_dir,
        #     num_to_save
        # )

    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": config["num_epochs"],
    }, results_dir / "checkpoint.pth")

    num_generated = config["num_generated"]
    all_generated = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        real_images, _ = next(iter(reference_dataloader))
        real_images = real_images.to(device)

        reconstructed = decoder(encoder(real_images))

        print("Input range:", real_images.min().item(), real_images.max().item())
        print("Output range:", reconstructed.min().item(), reconstructed.max().item())
        print("Input mean:", real_images.mean().item())
        print("Output mean:", reconstructed.mean().item())
        print("Input std:", real_images.std().item())
        print("Output std:", reconstructed.std().item())
        print("MSE:", nn.functional.mse_loss(
            reconstructed, real_images
        ).item())

        real_images = ((real_images + 1) / 2).clamp(0, 1)
        reconstructed = ((reconstructed + 1) / 2).clamp(0, 1)

    save_and_plot_generated(
        reconstructed.cpu(),
        real_images.cpu(),
        results_dir,
        len(real_images)
    )

    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device)
        print(f"Peak GPU memory: {peak_memory / 1024**3:.2f} GB")


    reference_images = []
    for images, _ in reference_dataloader:
        reference_images.append(images)

    reference_images = torch.cat(reference_images)
    reference_images = ((reference_images + 1) / 2).clamp(0, 1)

    compute_fid(real_images.cpu(), reconstructed.cpu())

    # save_and_plot_generated(generated, reference_images, results_dir, num_generated)
    plot_training_losses(
        reconstruction_losses,
        results_dir
    )