import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from demo_vae.modules_cifar10 import Encoder, Decoder, compute_fid, reparameterize
from demo_vae.utils_cifar10 import plot_training_losses, save_and_plot_generated
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

with CONFIG_PATH.open("r") as f:
    all_config = yaml.safe_load(f)

config = {
    **all_config["defaults"],
    **all_config["vae"],
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
    reconstruction_losses = []
    kl_losses = []
    criterion = nn.MSELoss()
    if device.type == "cuda":
        torch.cuda.synchronize()
        print(f"Training Timer started!")
    train_start = time.perf_counter()

    encoder = Encoder(latent_dim).to(device)
    decoder = Decoder(latent_dim).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config["learning_rate"],
    )

    beta = config.get("beta", 1.0)

    reconstruction_losses = []
    kl_losses = []
    total_losses = []
    for epoch in range(config["num_epochs"]):
        total_reconstruction_loss = 0.0
        total_kl_loss = 0.0
        total_loss = 0.0

        encoder.train()
        decoder.train()

        total_loss = 0.0

        for images, _ in train_dataloader:
            images = images.to(device)

            mu, logvar = encoder(images)
            z = reparameterize(mu, logvar)
            reconstructed = decoder(z)

            reconstruction_loss = F.mse_loss(
                reconstructed, images, reduction="sum"
            )

            kl_loss = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp()
            )

            loss = (reconstruction_loss + beta * kl_loss) / images.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_reconstruction_loss += reconstruction_loss.item()
            total_kl_loss += kl_loss.item()

        dataset_size = len(train_dataloader.dataset) #type:ignore
        total_losses.append(total_loss / dataset_size)
        reconstruction_losses.append(
            total_reconstruction_loss / dataset_size
        )
        kl_losses.append(total_kl_loss / dataset_size)
        print(
            f"Epoch [{epoch + 1}/{config['num_epochs']}] "
            f"Total: {total_losses[-1]:.4f} "
            f"Reconstruction: {reconstruction_losses[-1]:.4f} "
            f"KL: {kl_losses[-1]:.4f}"
        )

        torch.save({
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
        }, results_dir / "checkpoint.pth")
        
    num_generated = config["num_generated"]
    all_generated = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        print(f"Training time: {time.perf_counter() - train_start:.2f}s")
    
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        real_images, _ = next(iter(reference_dataloader))
        real_images = real_images.to(device)
        mu, logvar = encoder(real_images)
        reconstructed = decoder(mu)

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

    with torch.no_grad():
        z = torch.randn(
            num_generated, latent_dim, device=device
        )
        generated = decoder(z)

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

    compute_fid(generated.cpu(), reference_images.cpu())
    # save_and_plot_generated(generated, reference_images, results_dir, num_generated)
    plot_training_losses(
        reconstruction_losses, #type:ignore
        total_losses, #type:ignore
        results_dir
    )