# attentionhand_enhanced.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -------------------------------
# Adaptive / Learned Gaussian Filter
# -------------------------------
class AdaptiveGaussianFilter(nn.Module):
    """
    Applies a learned/adaptive Gaussian filter to attention maps.
    """
    def __init__(self, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.log_sigma = nn.Parameter(torch.zeros(1))  # learnable sigma

    def forward(self, attn_map):
        sigma = torch.exp(self.log_sigma)
        coords = torch.arange(self.kernel_size) - self.kernel_size // 2
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing='ij')
        kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2*sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.to(attn_map.device).unsqueeze(0).unsqueeze(0)
        attn_map = attn_map.unsqueeze(1)  # [B, 1, H, W]
        attn_map = F.conv2d(attn_map, kernel, padding=self.kernel_size//2)
        return attn_map.squeeze(1)

# -------------------------------
# Modern Autoencoder (SDXL-like)
# -------------------------------
class ModernAutoencoder(nn.Module):
    """
    Placeholder for SDXL-style encoder/decoder integration.
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

# Example simple encoder/decoder for placeholder
class SimpleEncoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, 4, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim=256, output_channels=3):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, output_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.deconv(z)

# -------------------------------
# Positional Encoding Utility
# -------------------------------
def positional_encoding(seq_len, dim, device='cpu'):
    pe = torch.zeros(seq_len, dim, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# -------------------------------
# VAS Guidance Network with Mesh Features
# -------------------------------
class VASGuidanceNetwork(nn.Module):
    """
    Guidance network enhanced with mesh features and positional encoding.
    """
    def __init__(self, input_dim=256, mesh_feat_dim=64, hidden_dim=512):
        super().__init__()
        self.mesh_feat_dim = mesh_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + mesh_feat_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  # output guidance
        )

    def forward(self, text_emb, mesh_feat, pos_enc):
        x = torch.cat([text_emb, mesh_feat, pos_enc], dim=-1)
        guidance = self.mlp(x)
        return guidance

# -------------------------------
# Example Forward Pass
# -------------------------------
if __name__ == "__main__":
    # Dummy attention map
    attn_map = torch.rand(2, 64, 64)  # [B, H, W]
    adaptive_filter = AdaptiveGaussianFilter(kernel_size=5)
    refined_attn = adaptive_filter(attn_map)
    print("Refined attention map shape:", refined_attn.shape)

    # Autoencoder forward
    encoder = SimpleEncoder()
    decoder = SimpleDecoder()
    autoencoder = ModernAutoencoder(encoder, decoder)
    images = torch.rand(2, 3, 64, 64)
    recon_images = autoencoder(images)
    print("Reconstructed image shape:", recon_images.shape)

    # VAS guidance network
    text_emb = torch.rand(2, 256)
    mesh_feat = torch.rand(2, 64)
    pos_enc = positional_encoding(seq_len=256, dim=256)
    vas_net = VASGuidanceNetwork(input_dim=256, mesh_feat_dim=64)
    guidance = vas_net(text_emb, mesh_feat, pos_enc)
    print("Guidance output shape:", guidance.shape)
