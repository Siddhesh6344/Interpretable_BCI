import torch
import datetime
import os
from dataloader import get_mnist_loaders
from train import train
from dense_autoencoder import DenseAutoencoder
from loss import top_k_auxiliary_loss
from postprocessing_scripts.analyze_dictionary_usage import analyze_dictionary_usage
from postprocessing_scripts.get_top_atoms import get_top_atoms
from postprocessing_scripts.plot_l0_l1_composite import plot_l0_l1_composite
from postprocessing_scripts.visualize_dict import visualize_selected_kernels
from postprocessing_scripts.visualize_dict import visualize_dictionary
import yaml

# ---------------------------
# Load Config
# ---------------------------
def load_config(path="pipeline/config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ---------------------------
# Main Function
# ---------------------------
def main():
    config = load_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DenseAutoencoder(
        in_channels=config['model']['in_channels'],
        num_kernels=config['model']['num_kernels'],
        kernel_size=config['model']['kernel_size'],
        k=config['model']['sparsity_k']
    ).to(device)

    train_loader, test_loader = get_mnist_loaders(batch_size=config['training']['batch_size'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    train(model, train_loader, optimizer, top_k_auxiliary_loss, device, num_epochs=config['training']['num_epochs'])

    # Analyze usage across training set
    l0_counts, l1_sums = analyze_dictionary_usage(model, train_loader, device)
    top_l0, top_l1 = get_top_atoms(l0_counts, l1_sums, top_k=config['postprocessing']['top_k_visualization'])

    if config['postprocessing']['plot_l0_l1_composite']:
        plot_l0_l1_composite(model.decoder.weight, top_l0, top_l1, title=" Dictionary Usage")

    if 'l0' in config['postprocessing']['visualize_atoms_by']:
        visualize_selected_kernels(model.decoder.weight, top_l0, title="Top Atoms by L0")

    if 'l1' in config['postprocessing']['visualize_atoms_by']:
        visualize_selected_kernels(model.decoder.weight, top_l1, title="Top Atoms by L1")

    visualize_dictionary(model.decoder.weight, num_kernels=config['postprocessing']['dictionary_visualization_count'])

    timestamp = datetime.datetime.now().strftime(config['saving']['timestamp_format'])
    save_dir = os.path.join(config['saving']['root_dir'], timestamp)
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "trained_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# ---------------------------
# Run It
# ---------------------------
if __name__ == '__main__':
    main()
