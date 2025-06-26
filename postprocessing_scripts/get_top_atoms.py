import torch

def get_top_atoms(l0_counts, l1_sums, top_k=1000):
    top_l0 = torch.topk(l0_counts, top_k).indices
    top_l1 = torch.topk(l1_sums, top_k).indices
    return top_l0, top_l1
