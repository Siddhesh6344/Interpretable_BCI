import os
import matplotlib.pyplot as plt

def visualize_codes_conv(z_sparse, save_path="./Figures and results/activation_vs_freq.png"):
    """
    Visualizes average activation vs activation frequency (Figure 4 style).

    Args:
        z_sparse (torch.Tensor): Sparse code of shape [B, C, H, W]
        save_path (str): Path to save the figure
    """
    avg_act = z_sparse.mean(dim=[0, 2, 3])          # (C,)
    freq = (z_sparse != 0).float().sum(dim=[0, 2, 3])  # (C,)

    plt.figure(figsize=(6, 4))
    plt.scatter(freq.cpu(), avg_act.cpu(), s=10)
    plt.xlabel("Activation Frequency")
    plt.ylabel("Average Activation")
    plt.title("Figure 4: Avg Activation vs Frequency")
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved activation vs frequency plot to {save_path}")
