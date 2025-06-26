import torch

def analyze_dictionary_usage(model, dataloader, device):
    model.eval()
    num_channels = model.encoder.out_channels
    l0_counts = torch.zeros(num_channels).to(device)
    l1_sums = torch.zeros(num_channels).to(device)

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            _, pre_codes, codes, _ = model(x)

            # Flatten codes: (B, C, H, W) → (C, B*H*W)
            codes_flat = codes.view(codes.shape[0], codes.shape[1], -1)

            # ℓ₀: Count non-zero activations per channel
            l0_counts += (codes_flat != 0).sum(dim=(0, 2))

            # ℓ₁: Sum of absolute activations
            l1_sums += codes_flat.abs().sum(dim=(0, 2))

    return l0_counts.cpu(), l1_sums.cpu()