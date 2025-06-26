import os
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# ----------------------
# Clustering Functions
# ----------------------

def cluster_images(images, n_clusters=10):
    """
    Clusters a batch of images using KMeans.

    Args:
        images (torch.Tensor): Shape (B, C, H, W)
        n_clusters (int): Number of clusters

    Returns:
        labels (np.ndarray): Cluster labels
    """
    flat = images.view(images.shape[0], -1).cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(flat)
    return labels


def cluster_atoms(dictionary, method='umap', save_path='./Figures and results/cluster_atoms.png'):
    """
    Reduces dimensionality of atoms and saves scatter plot.

    Args:
        dictionary (torch.Tensor): Shape (N, C, H, W)
        method (str): 'umap' or 'tsne'
        save_path (str): Filepath to save plot
    """
    D = dictionary.view(dictionary.size(0), -1).cpu().numpy()
    reducer = umap.UMAP() if method == 'umap' else TSNE()
    embed = reducer.fit_transform(D)

    plt.figure(figsize=(6, 6))
    plt.scatter(embed[:, 0], embed[:, 1], cmap='Spectral')
    plt.title("Clustering of Atoms")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved atom clustering plot to {save_path}")


def cluster_codes(z_sparse, method='umap', save_path='./Figures and results/cluster_codes.png'):
    """
    Reduces dimensionality of sparse codes and saves scatter plot.

    Args:
        z_sparse (torch.Tensor): Shape (B, C, H, W)
        method (str): 'umap' or 'tsne'
        save_path (str): Filepath to save plot
    """
    B, C, H, W = z_sparse.shape
    flat_codes = z_sparse.view(B, -1).cpu().numpy()
    reducer = umap.UMAP() if method == 'umap' else TSNE()
    embed = reducer.fit_transform(flat_codes)

    plt.figure(figsize=(6, 6))
    plt.scatter(embed[:, 0], embed[:, 1], cmap='Spectral')
    plt.title("Clustering of Code Distributions")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved code clustering plot to {save_path}")