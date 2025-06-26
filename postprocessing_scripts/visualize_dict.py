import matplotlib.pyplot as plt
import os

def visualize_dictionary(dictionary, num_kernels=1000, cols=10, save_path='./Figures and results/dictionary.png'):
    dictionary = dictionary.detach().cpu()
    kernels = dictionary.squeeze(0)  # shape: (num_kernels, H, W)

    rows = num_kernels // cols + int(num_kernels % cols != 0)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    axes = axes.flatten()

    for i in range(num_kernels):
        ax = axes[i]
        kernel = kernels[i].squeeze().numpy()
        ax.imshow(kernel, cmap='gray')
        ax.axis('off')
        ax.set_title(f'{i}', fontsize=6)

    for j in range(num_kernels, len(axes)):
        axes[j].axis('off')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved full dictionary grid to {save_path}")

def visualize_selected_kernels(dictionary, selected_indices, save_dir='./Figures and results/selected_kernels',title="Top Atoms by L0"):
    dictionary = dictionary.detach().cpu()
    kernels = dictionary.squeeze(0)  # (C, H, W)

    os.makedirs(save_dir, exist_ok=True)

    for idx in selected_indices:
        kernel = kernels[idx].squeeze().numpy()
        plt.figure(figsize=(2, 2))
        plt.imshow(kernel, cmap='gray')
        plt.axis('off')
        plt.title(f'{idx}')
        save_path = os.path.join(save_dir, f'kernel_{idx}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved kernel {idx} to {save_path}")
