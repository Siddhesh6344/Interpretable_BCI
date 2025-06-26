import torch
import torch.nn.functional as F

def top_k_auxiliary_loss(x, x_hat, pre_codes, codes, dictionary, penalty=0.1):
    """
    x:         [B, 1, 28, 28]
    x_hat:     [B, 1, 28, 28]
    pre_codes: [B, C, 1, 1]
    codes:     [B, C, 1, 1]
    dictionary: [C, 1, 28, 28]
    """
    residual = x - x_hat
    mse = residual.square().mean()

    # Flatten
    B, C, _, _ = codes.shape
    pre_codes_flat = torch.relu(pre_codes.view(B, C))       # [B, C]
    codes_flat = codes.view(B, C)                           # [B, C]
    residual_codes = pre_codes_flat - codes_flat            # [B, C]

    # Auxiliary top-k
    k_aux = residual_codes.shape[1] // 2
    topk_vals, topk_indices = torch.topk(residual_codes, k=k_aux, dim=1)
    aux_codes = torch.zeros_like(residual_codes).scatter(1, topk_indices, topk_vals)  # [B, C]

    # Reconstruct using decoder dictionary
    dictionary_flat = dictionary.view(C, -1)  # [C, 784]
    aux_recon_flat = aux_codes @ dictionary_flat  # [B, 784]
    aux_recon = aux_recon_flat.view(B, 1, 28, 28)

    aux_mse = (residual - aux_recon).square().mean()

    return mse + penalty * aux_mse
