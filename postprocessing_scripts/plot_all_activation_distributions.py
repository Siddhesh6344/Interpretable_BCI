import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def plot_activation_distribution(z_sparse, model_name="TopK", color='blue'):
    """
    Plots a single model's frequency vs average activation as a KDE contour.

    Args:
        z_sparse (torch.Tensor): Shape [B, C, H, W]
        model_name (str): Label for legend
        color (str): Color for plot
    """
    z = z_sparse.detach().cpu()
    B, C, H, W = z.shape
    z = z.view(B, C, -1)  # [B, C, H*W]

    freq = (z != 0).sum(dim=2).float().mean(dim=0).numpy()
    avg = z.abs().sum(dim=2).float().mean(dim=0).numpy() / z.shape[2]

    sns.kdeplot(
        x=freq,
        y=avg,
        fill=False,
        levels=5,
        bw_adjust=0.5,
        thresh=0.01,
        label=model_name,
        color=color,
        linewidths=1.5
    )


def plot_all_activation_distributions(model_z_dict, save_path="./Figures and results/activation_distribution_kde.png"):
    """
    Plots KDE contours of activation frequency vs average activation for multiple models.

    Args:
        model_z_dict (dict): {model_name: (z_sparse_tensor, color)}
        save_path (str): Path to save the final plot
    """
    plt.figure(figsize=(8, 6))
    for name, (z_sparse, color) in model_z_dict.items():
        plot_activation_distribution(z_sparse, model_name=name, color=color)

    plt.xlabel("Activation Frequency")
    plt.ylabel("Average Activation")
    plt.title("Figure 4(a): Frequency vs Avg Activation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved activation distribution KDE to {save_path}")
