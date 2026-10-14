import torch
import matplotlib.pyplot as plt
import time
from torchvision.utils import save_image



def save_generated_samples(G, epoch, latent_dim, device, results_dir, batch_size):
    # epoch-level visualization of generated samples
    G.eval()

    with torch.no_grad():
        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        generated = G(z).cpu()

    G.train()

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))

    for i, ax in enumerate(axes.flat):
        ax.imshow(generated[i].permute(1, 2, 0) * 0.5 + 0.5)
        ax.axis("off")

    fig.savefig(
        results_dir / f"epoch_{epoch:02d}.png",
        bbox_inches="tight"
    )
    plt.close(fig)

def save_and_plot_generated(generated, reference_images, results_dir, num_to_save):
    
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
        f"{results_dir}/generated_cifar10.png",
        nrow=8,
        normalize=False
    )

    save_image(
        reference_images[:min(num_to_save, reference_images.size(0))],
        f"{results_dir}/reference_cifar10.png",
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
