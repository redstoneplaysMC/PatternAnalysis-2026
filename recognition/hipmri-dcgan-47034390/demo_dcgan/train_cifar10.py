import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from pathlib import Path
from modules_cifar10 import Generator, Discriminator, compute_fid
import yaml
from utils_cifar10 import save_generated_samples, save_and_plot_generated, plot_training_losses

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

with CONFIG_PATH.open("r") as f:
    all_config = yaml.safe_load(f)

config = {
    **all_config["defaults"],
    **all_config["dcgan"],
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

print(images.shape)
print(labels.shape)
print(images.min())
print(images.max())


if __name__ == "__main__":

    channels=3
    image_size=32
    
    G = Generator(latent_dim, channels=channels, image_size=image_size).to(device)
    D = Discriminator(channels=channels, image_size=image_size).to(device)

    generator_losses = []
    discriminator_losses = []

    criterion = nn.BCEWithLogitsLoss()

    optimizer_G = torch.optim.Adam(
        G.parameters(),
        lr=config["learning_rate_G"],
        betas=tuple(config["betas"])
    )

    optimizer_D = torch.optim.Adam(
        D.parameters(),
        lr=config["learning_rate_D"],
        betas=tuple(config["betas"])
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"Training Timer started!")
    train_start = time.perf_counter()

    # Training loop for the GAN
    for epoch in range(config["num_epochs"]):
        
        for real_images, _ in train_dataloader: # Iterate over batches of real images from the dataloader, depending on batch size
            real_images = real_images.to(device) # Move real images to the specified device (CPU or GPU)
            batch_size = real_images.size(0) # Get the batch size of the current set of real images
            real_labels = torch.ones(batch_size, device=device) # Labels for real images
            fake_labels = torch.zeros(batch_size, device=device) # Labels for fake images

            # Train Discriminator
            optimizer_D.zero_grad() # Zero the gradients for the discriminator
            real_output = D(real_images) # Get the discriminator's output for real images
            loss_real = criterion(real_output, real_labels) # Compute the loss for real images
            z = torch.randn(batch_size, latent_dim, 1, 1, device=device) # Sample random noise for the generator
            fake_images = G(z) # Generate fake images using the generator
            fake_output = D(fake_images.detach()) # Get the discriminator's output for fake images
            loss_fake = criterion(fake_output, fake_labels) # Compute the loss for fake images
            loss_D = loss_real + loss_fake # Total discriminator loss
            loss_D.backward() # Backpropagate the discriminator loss
            optimizer_D.step() # Update the discriminator's parameters

            # Train Generator
            optimizer_G.zero_grad() # Zero the gradients for the generator
            fake_output = D(fake_images) # Get the discriminator's output for the fake images
            loss_G = criterion(fake_output, real_labels) # Compute the generator's loss
            loss_G.backward() # Backpropagate the generator's loss
            optimizer_G.step() # Update the generator's parameters
            generator_losses.append(loss_G.item()) # Record the generator's loss
            discriminator_losses.append(loss_D.item()) # Record the discriminator's loss
            
        if epoch + 1 in [5, 10, 15, 20]:
            save_generated_samples(
                G,
                epoch + 1,
                latent_dim,
                device,
                results_dir,
                config["batch_size"]
            )
    
        print(
            f"Epoch [{epoch + 1}/{config['num_epochs']}] "
            f"Loss D: {loss_D.item():.4f} " #type:ignore
            f"Loss G: {loss_G.item():.4f}" #type:ignore
        )
        generator_losses.append(loss_G.item()) #type:ignore
        discriminator_losses.append(loss_D.item()) #type:ignore
        

    torch.save({
        "generator": G.state_dict(),
        "discriminator": D.state_dict(),
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D": optimizer_D.state_dict(),
        "epoch": config["num_epochs"],
    }, results_dir / "checkpoint.pth")

    # Save generated images and comparison plot.
    generated_dir = results_dir / "generated-images"
    generated_dir.mkdir(parents=True, exist_ok=True)

    G.eval()

    num_generated = 16
    all_generated = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for start in range(0, num_generated, config["batch_size"]):
            batch_size = min(
                config["batch_size"],
                num_generated - start
            )

            z = torch.randn(
                batch_size, latent_dim, 1, 1, device=device
            )

            all_generated.append(G(z).cpu())

    generated = torch.cat(all_generated)
    generated = ((generated + 1) / 2).clamp(0, 1)

    # Get 16 reference images.
    reference_images, _ = next(iter(reference_dataloader))
    reference_images = reference_images[:16]
    reference_images = ((reference_images + 1) / 2).clamp(0, 1)

    # Plot 16 reference images on the left and 16 generated images on the right.
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))

    for i in range(16):
        row = i // 4
        col = i % 4

        axes[row, col].imshow(
            reference_images[i].permute(1, 2, 0)
        )
        axes[row, col].axis("off")

        axes[row, col + 4].imshow(
            generated[i].permute(1, 2, 0)
        )
        axes[row, col + 4].axis("off")

    axes[0, 1].set_title("Reference", pad=12)
    axes[0, 5].set_title("Generated (DC-GAN)", pad=12)

    plt.tight_layout()
    plt.savefig(
        generated_dir / "reference_vs_generated.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.show()
    plt.close(fig)

    # Save generated images individually as a grid.
    save_image(
        generated,
        generated_dir / "generated_grid.png",
        nrow=4
    )

    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device)
        print(f"Peak GPU memory: {peak_memory / 1024**3:.2f} GB")

    compute_fid(reference_images, generated)
    plot_training_losses(generator_losses, discriminator_losses, results_dir)