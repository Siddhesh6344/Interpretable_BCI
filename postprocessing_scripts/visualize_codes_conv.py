import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_conv_atoms(conv_weights, num_atoms=16, title="Conv Atoms", save_path="./Figures and results/conv_atoms.png"):
    """
    Visualizes a grid of convolutional atoms (filters).

    Args:
        conv_weights (torch.Tensor): Shape [C_out, 1, H, W]
        num_atoms (int): Number of filters to visualize
        title (str): Plot title
        save_path (str): Where to save the plot
    """
    weights = conv_weights[:num_atoms].cpu().detach()
    weights = weights.squeeze(1)  # assume grayscale: [C_out, 1, H, W]\
    n_cols = min(num_atoms, 10)
    n_rows = int(np.ceil(num_atoms / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    axs = axs.flatten()
    for i in range(num_atoms):
        axs[i].imshow(weights[i], cmap='gray')
        axs[i].axis('off')
    for j in range(num_atoms, len(axs)):
        axs[j].axis('off')
    plt.suptitle(title)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved conv atoms visualization to {save_path}")
