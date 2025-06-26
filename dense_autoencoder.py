import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseAutoencoder(nn.Module):
    def __init__(self, in_channels=1, num_kernels=1000, kernel_size=28, k=10, use_sparsity=True):
        super().__init__()
        self.k = k
        self.use_sparsity = use_sparsity

        self.encoder = nn.Conv2d(in_channels, num_kernels, kernel_size=kernel_size, stride=1, padding=0)  # [B, C, 1, 1]
        self.decoder = nn.ConvTranspose2d(num_kernels, in_channels, kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        pre_codes = self.encoder(x)  # [B, 1000, 1, 1]
        pre_codes = pre_codes.view(pre_codes.size(0), pre_codes.size(1))  # [B, 1000]

        if self.use_sparsity:
            topk_vals, _ = torch.topk(torch.abs(pre_codes), self.k, dim=1)
            threshold = topk_vals[:, -1].unsqueeze(-1)  # [B, 1]
            codes = torch.where(torch.abs(pre_codes) >= threshold, pre_codes, torch.zeros_like(pre_codes))
        else:
            codes = pre_codes  # Dense setting

        codes = codes.view(codes.size(0), codes.size(1), 1, 1)  # [B, C, 1, 1]
        x_hat = self.decoder(codes)  # [B, 1, 28, 28]
        return x_hat, pre_codes.view(-1, pre_codes.size(1), 1, 1), codes, self.decoder.weight