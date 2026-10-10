import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchmetrics.functional.image import structural_similarity_index_measure
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from pathlib import Path
from modules_mnist import Generator, Discriminator, compute_fid
import yaml
    
# https://medium.com/@masonthecount/using-a-dcgan-to-learn-the-mnist-dataset-in-order-to-generate-images-1ca616e45463
# AI-generated code.

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

results_dir = Path(config["results_dir"])
results_dir.mkdir(parents=True, exist_ok=True)
num_to_save = max(64, config["num_generated"])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = Subset(
    datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    ),
    range(config["size_train_dataset"])
)

train_dataloader = DataLoader(
    # Create a DataLoader for the training dataset with the specified batch size and shuffling
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True
)


reference_dataset = Subset(
    datasets.MNIST(
        # Create the reference dataset for evaluation purposes
        root="./data",
        train=False,
        download=True,
        transform=transform
    ),
    range(config["size_reference_dataset"])
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

def save_generated_samples(G, epoch, latent_dim, device, results_dir, batch_size):
    # epoch-level visualization of generated samples
    G.eval()

    with torch.no_grad():
        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        generated = G(z).cpu()

    G.train()

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))

    for i, ax in enumerate(axes.flat):
        ax.imshow(generated[i, 0], cmap="gray")
        ax.axis("off")

    fig.savefig(
        results_dir / f"epoch_{epoch:02d}.png",
        bbox_inches="tight"
    )
    plt.close(fig)


def save_and_plot_generated(generated, reference_images, results_dir):
    
    torch.save(
        generated,
        results_dir / "generated_images.pt"
    )

    torch.save(
        reference_images,
        results_dir / "reference_images.pt"
    )
    
    save_image(
        generated[:min(num_to_save, generated.size(0))],
        f"{results_dir}/generated_mnist.png",
        nrow=8,
        normalize=False
    )

    save_image(
        reference_images[:min(num_to_save, reference_images.size(0))],
        f"{results_dir}/reference_mnist.png",
        nrow=8,
        normalize=False
    )

def plot_training_losses(generator_losses, discriminator_losses, results_dir):
    # PLOTTING
    plt.figure()
    plt.plot(
        generator_losses,
        label="Generator"
    )

    plt.plot(
        discriminator_losses,
        label="Discriminator"
    )

    plt.xlabel("Training iteration")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(
        f"{results_dir}/training_losses.png",
        dpi=150,
        bbox_inches="tight"
    )

    fig, ax = plt.subplots()

    ax.plot(generator_losses, label="Generator")
    ax.plot(discriminator_losses, label="Discriminator")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("DCGAN Training Loss")
    ax.legend()

    fig.savefig(
        results_dir / "training_losses.png",
        bbox_inches="tight"
    )

    plt.close(fig)
    
def generate_images(G, latent_dim, device):
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        z = torch.randn(16, latent_dim, 1, 1, device=device)
        generated = G(z)

    if device.type == "cuda":
        torch.cuda.synchronize()

    generation_time = time.perf_counter() - start

    return generated, generation_time

if __name__ == "__main__":

    G = Generator(latent_dim).to(device)
    D = Discriminator().to(device)

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

    G = Generator(latent_dim).to(device)
    G.eval()

    # This generates num_generated fake images in batches, rather than generating all of them at once.
    num_generated = config["num_generated"]

    all_generated = []

    with torch.no_grad():

        for _ in range(0, num_generated, config["batch_size"]):

            batch_size = min(
                config["batch_size"],
                num_generated - len(all_generated)
            )

            z = torch.randn(
                batch_size,
                latent_dim,
                1,
                1,
                device=device
            )

            fake = G(z)

            all_generated.append(fake.cpu())

    generated = torch.cat(all_generated)

    if device.type == "cuda":

        peak_memory = torch.cuda.max_memory_allocated()

        print(
            f"Peak GPU memory: "
            f"{peak_memory / 1024**3:.2f} GB"
        )

    #  Normalize generated images to the range [0, 1] for FID computation.
    generated = (generated + 1) / 2
    reference_images = []

    for images, _ in reference_dataloader:
        reference_images.append(images)

    reference_images = torch.cat(reference_images)
    reference_images = (reference_images + 1) / 2

    compute_fid(reference_images, generated)
    
    #  Set the generator to evaluation mode before generating new images.
    G.eval()

    generated, generation_time = generate_images(G, latent_dim, device)
    print(f"Generation time: {generation_time:.4f} seconds")

    save_and_plot_generated(generated, reference_images, results_dir)
    plot_training_losses(generator_losses, discriminator_losses, results_dir)