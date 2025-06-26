import os
import matplotlib.pyplot as plt

def plot_l0_l1_composite(dictionary, top_l0, top_l1, title='Feature Selection vs Activation Levels', save_path="./Figures and results/l0_l1_composite.png"):
    """
    Visualizes dictionary atoms selected by top-ℓ₀ and top-ℓ₁ stats in two rows, and saves the plot.

    Args:
        dictionary (torch.Tensor): Shape (1, C, H, W) from ConvTranspose2d
        top_l0 (torch.Tensor or list): Indices of top-k atoms based on ℓ₀ (activation frequency)
        top_l1 (torch.Tensor or list): Indices of top-k atoms based on ℓ₁ (activation magnitude)
        title (str): Title of the figure
        save_path (str): Path to save the figure
    """
    dictionary = dictionary.detach().cpu().squeeze(0)  # (C, H, W)
    num_kernels = len(top_l0)
    fig, axes = plt.subplots(2, num_kernels, figsize=(num_kernels * 1.5, 3))
    fig.suptitle(title, fontsize=16)

    for i in range(num_kernels):
        # Top row: ℓ₀
        ax_top = axes[0, i]
        kernel_l0 = dictionary[top_l0[i]].squeeze().numpy()
        ax_top.imshow(kernel_l0, cmap='gray')
        ax_top.axis('off')
        if i == 0:
            ax_top.set_ylabel(r'Top $\ell_0$', fontsize=12)

        # Bottom row: ℓ₁
        ax_bot = axes[1, i]
        kernel_l1 = dictionary[top_l1[i]].squeeze().numpy()
        ax_bot.imshow(kernel_l1, cmap='gray')
        ax_bot.axis('off')
        if i == 0:
            ax_bot.set_ylabel(r'Top $\ell_1$', fontsize=12)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved ℓ₀/ℓ₁ composite visualization to {save_path}")
