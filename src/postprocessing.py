import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import datetime
import os
import yaml
from dataloader import get_mnist_loaders
from train import train
from loss import top_k_auxiliary_loss
from dense_autoencoder import DenseAutoencoder
from postprocessing_scripts.compute_babel_score import babel_score
from postprocessing_scripts.compute_babel_score import babel_score_for_sample
from postprocessing_scripts.analyze_dictionary_usage import analyze_dictionary_usage
from postprocessing_scripts.get_top_atoms import get_top_atoms
from postprocessing_scripts.plot_l0_l1_composite import plot_l0_l1_composite
from postprocessing_scripts.visualize_dict import visualize_selected_kernels
from postprocessing_scripts.visualize_dict import visualize_dictionary
from postprocessing_scripts.visualize_codes_conv import visualize_conv_atoms
from postprocessing_scripts.sort_atoms_by_l0 import sort_atoms_by_l0
from postprocessing_scripts.sort_atoms_by_l1 import sort_atoms_by_l1
from postprocessing_scripts.sequential_reconstruction import sequential_reconstruction
from postprocessing_scripts.clustering import cluster_atoms
from postprocessing_scripts.clustering import cluster_codes
from postprocessing_scripts.clustering import cluster_images
from postprocessing_scripts.visualize_sorted_atoms import visualize_sorted_atoms
from postprocessing_scripts.visualize_code_activations import visualize_codes_conv
from postprocessing_scripts.plot_all_activation_distributions import plot_all_activation_distributions
from postprocessing_scripts.plot_all_activation_distributions import plot_activation_distribution


# ----------------------
# Load config
# ----------------------
def load_config(path="pipeline/postprocessing_config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ----------------------
# Load one MNIST batch
# ----------------------

def get_mnist_batch(batch_size=128):
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return next(iter(test_loader))

# ----------------------
# Run Postprocessing
# ----------------------
def run_postprocessing():
    config = load_config()
    c = config['postprocessing']

    device = torch.device("cuda" if c['use_cuda'] and torch.cuda.is_available() else "cpu")

    model = DenseAutoencoder(
        in_channels=1,
        num_kernels=1000,
        kernel_size=28,
        k=c['sparsity_k'],
        use_sparsity=True
    )
    model.load_state_dict(torch.load(c['model_path'], map_location=device))
    model.to(device).eval()

    x, _ = get_mnist_batch(batch_size=c['dataset']['batch_size'])
    x = x.to(device)

    with torch.no_grad():
        x_recon, pre_codes, codes, dictionary = model(x)
        z_sparse = codes

    output_dir = c['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # if c['visualizations']['visualize_conv_filters']:
    #     visualize_conv_atoms(model.encoder.weight, title="Learned Conv Filters")

    if c['visualizations']['sort_by_l0']:
        l0_scores = sort_atoms_by_l0(z_sparse)
        visualize_sorted_atoms(model.decoder.weight, l0_scores, title="Atoms Sorted by L0")

    if c['visualizations']['sort_by_l1']:
        l1_scores = sort_atoms_by_l1(z_sparse)
        visualize_sorted_atoms(model.decoder.weight, l1_scores, title="Atoms Sorted by L1")

    # if c['visualizations']['visualize_codes_conv']:
    #     visualize_codes_conv(z_sparse)

    if c['visualizations']['plot_activation_distributions']:
        plot_all_activation_distributions(
            {"TopK": (z_sparse, 'blue')},
            save_path=os.path.join(output_dir, "activation_distribution_kde.png")
        )

    if c['metrics']['compute_babel_score']:
        global_babel = babel_score(model.decoder.weight)
        print(f"Global Babel Score: {global_babel:.4f}")
    
    if c['metrics']['compute_sample_babel_score']:
        per_sample_babel = babel_score_for_sample(z_sparse, model.decoder.weight)
        print(f"Per-Sample Babel Score (mean): {per_sample_babel:.4f}")

    if c['visualizations']['sequential_reconstruction']:
        sequential_reconstruction(x[0:1], z_sparse[0:1], model.decoder, top_k=c['reconstruction']['top_k'])

    if c['visualizations']['cluster_atoms']:
        cluster_atoms(model.decoder.weight.detach())

    if c['visualizations']['cluster_codes']:
        cluster_codes(z_sparse)

# ----------------------
# Entry Point
# ----------------------
if __name__ == "__main__":
    run_postprocessing()
