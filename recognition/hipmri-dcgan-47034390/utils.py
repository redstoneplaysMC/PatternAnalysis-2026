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
    

"""Init Weights for the network layers
Parameters: model - the neural network model whose weights need to be initialized
"""

"""Compute SSIM
Parameters: img1 - first image
            img2 - second image
"""
import torch.nn.functional as F

def compute_ssim(img1, img2, window_size=11, size_average=True):
    """
    Code was copied from samuel2003-coder.
    Compute Structured Similarity Index (SSIM) between two images
    
    Args:
        img1, img2: Images in range [-1, 1] or [0, 1]
        window_size: Size of the Gaussian window
        size_average: Whether to average the SSIM over the batch
    Returns:
        SSIM value
    """

    # Constants for stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor([
        torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
        for x in range(window_size)
    ])
    window = gauss / gauss.sum()
    window = window.unsqueeze(1)
    window = window.mm(window.t()).float().unsqueeze(0).unsqueeze(0)
    window = window.to(img1.device)
    
    # Ensure images are in [0, 1] range
    if img1.min() < 0:
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2
    
    # Calculate means
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=1)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=1)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=1) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=1) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map

# def normalize(tensor):
#     """
#     Normalize tensor from [0, 1] to [-1, 1]
    
#     Args:
#         tensor: Image tensor in range [0, 1]
#     Returns:
#         Normalized tensor in range [-1, 1]
#     """
#     return tensor * 2 - 1

# # Converting images back too 0,1 range so they may be saved and displayed correctly
# def denormalize(tensor):
#     """
#     Denormalize tensor from [-1, 1] to [0, 1]
    
#     Args:
#         tensor: Image tensor in range [-1, 1]
#     Returns:
#         Denormalized tensor in range [0, 1]
#     """
#     return (tensor + 1) / 2