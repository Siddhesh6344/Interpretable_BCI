import torch
from postprocessing_scripts.visualize_atoms import visualize_atoms

def visualize_sorted_atoms(filters, scores, title):
    sorted_idx = torch.argsort(scores, descending=True)
    filters = filters[sorted_idx]
    visualize_atoms(filters, title)