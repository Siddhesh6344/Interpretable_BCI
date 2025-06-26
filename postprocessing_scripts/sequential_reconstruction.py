import torch
import matplotlib.pyplot as plt
import os

def sequential_reconstruction(image, z_sparse, decoder, top_k, save_path="./Figures and results/sequential_reconstruction.png"):
    """
    Visualizes sequential reconstruction as atoms are added one by one.

    Args:
        image (torch.Tensor): Original image tensor (unused in current plot, can be added later).
        z_sparse (torch.Tensor): Sparse code tensor of shape [1, C, H, W].
        decoder (nn.Module): Decoder model.
        top_k (int): Number of top atoms to consider.
        save_path (str): Path to save the reconstruction figure.
    """
    C = z_sparse.shape[1]
    flat = z_sparse.view(C, -1)
    l1_scores = torch.norm(flat, p=1, dim=1)
    topk_indices = torch.topk(l1_scores, top_k).indices

    reconstructions = []
    used_indices = []

    for k in range(1, top_k + 1):
        mask = torch.zeros_like(z_sparse)
        for idx in topk_indices[:k]:
            mask[:, idx] = z_sparse[:, idx]
        recon = decoder(mask)
        reconstructions.append(recon[0, 0].detach().cpu().numpy())
        used_indices.append(topk_indices[:k].cpu().numpy())

    # Plotting
    fig, axes = plt.subplots(2, top_k, figsize=(2 * top_k, 4))

    for i in range(top_k):
        axes[0, i].set_title(f"Atom {i+1}", fontsize=8)
        axes[0, i].imshow(used_indices[i].reshape(1, -1), cmap='viridis', aspect='auto')
        axes[1, i].imshow(reconstructions[i], cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].axis('off')

    plt.suptitle("Figure 6: Sequential Reconstruction", fontsize=12)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sequential reconstruction figure to {save_path}")