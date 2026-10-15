import torch
import matplotlib.pyplot as plt

def save_generated_samples(G, epoch, latent_dim, device, results_dir, batch_size):
    G.eval()

    with torch.no_grad():
        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        generated = G(z).cpu()

    G.train()

    fig, axes = plt.subplots(
        4, 4,
        figsize=(6, 6),
        gridspec_kw={"wspace": 0, "hspace": 0}
    )

    for i, ax in enumerate(axes.flat):
        ax.imshow(
            generated[i, 0] * 0.5 + 0.5,
            cmap="gray",
            vmin=0,
            vmax=1
        )
        ax.axis("off")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(
        results_dir / f"epoch_{epoch:02d}.png",
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close(fig)

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