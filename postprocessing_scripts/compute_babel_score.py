import torch
import torch.nn.functional as F
import numpy as np

# ----------------------
# Coherence (Babel Score)
# ----------------------
def babel_score(dictionary):
    D = dictionary.view(dictionary.size(0), -1)  # (n_kernels, flattened)
    D = F.normalize(D, dim=1)
    G = torch.matmul(D, D.T)  # Gram matrix
    mask = torch.eye(G.shape[0], device=G.device).bool()
    G.masked_fill_(mask, -1.0)
    return G.max(dim=1).values.mean().item()

# ----------------------
# Babel Score for selected atoms
# ----------------------
def babel_score_for_sample(z_sparse, dictionary):
    B, C, H, W = z_sparse.shape
    scores = []
    for i in range(B):
        active = (z_sparse[i] != 0).view(C, -1).sum(dim=1) > 0
        active_dict = dictionary[active]
        if active_dict.shape[0] > 1:
            scores.append(babel_score(active_dict))
    return np.mean(scores)
