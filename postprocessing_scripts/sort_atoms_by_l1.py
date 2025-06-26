import torch

def sort_atoms_by_l1(z_sparse):
    l1_scores = torch.abs(z_sparse).sum(dim=[0, 2, 3])
    return l1_scores
