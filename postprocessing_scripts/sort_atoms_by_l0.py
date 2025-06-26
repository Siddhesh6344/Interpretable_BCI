def sort_atoms_by_l0(z_sparse):
    l0_scores = (z_sparse != 0).float().sum(dim=[0, 2, 3])
    return l0_scores

