# # ----------------------
# # Visualize Atoms (Conv Filters)
# # ----------------------
# import torch
# import matplotlib.pyplot as plt
# import os

# def visualize_atoms(filters, title="Learned Atoms", save_path="./Figures and results/learned_atoms.png"):
#     filters = filters.detach().cpu().numpy()
#     n = filters.shape[0]
#     fig, axes = plt.subplots(1, n, figsize=(n * 1.5, 2))
#     if n == 1:
#         axes = [axes]  # in case of single filter

#     for i, ax in enumerate(axes):
#         ax.imshow(filters[i].squeeze(), cmap='gray')
#         ax.axis('off')

#     plt.suptitle(title)
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"Saved learned atoms visualization to {save_path}")


import torch
import matplotlib.pyplot as plt
import os
import numpy as np

def visualize_atoms(filters, title="Learned Atoms", save_path="./Figures and results/learned_atoms.png", cols=20):
    """
    Visualize dictionary atoms (conv filters) in a grid layout and save the figure.

    Args:
        filters (torch.Tensor): Tensor of shape (C, 1, H, W) or (C, H, W)
        title (str): Title of the figure
        save_path (str): Path to save the figure
        cols (int): Number of columns in the figure grid
    """
    filters = filters.detach().cpu().numpy()
    if filters.ndim == 4:  # (C, 1, H, W)
        filters = filters.squeeze(1)
    
    n = filters.shape[0]
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = axes.flatten()

    for i in range(n):
        axes[i].imshow(filters[i], cmap='gray')
        axes[i].axis('off')
    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.suptitle(title)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved learned atoms visualization to {save_path}")
