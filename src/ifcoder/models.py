import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """
    Simple conv VAE for patches shaped (C, H, W).
    Assumes H and W are divisible by 4 (two stride-2 downsamples).
    """

    def __init__(self, in_channels: int, patch_size: int, latent_dim: int = 16, hidden: int = 32):
        super().__init__()
        if patch_size % 4 != 0:
            raise ValueError("patch_size must be divisible by 4 (e.g., 16, 32, 64).")

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.latent_dim = latent_dim

        # Encoder: (C,H,W) -> (hidden*2, H/4, W/4)
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, stride=2, padding=1),  # /2
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden * 2, kernel_size=3, stride=2, padding=1),   # /4
            nn.ReLU(inplace=True),
        )

        h4 = patch_size // 4
        enc_feat = (hidden * 2) * h4 * h4
        self.fc_mu = nn.Linear(enc_feat, latent_dim)
        self.fc_logvar = nn.Linear(enc_feat, latent_dim)

        # Decoder: z -> (hidden*2, H/4, W/4) -> (C,H,W)
        self.fc_z = nn.Linear(latent_dim, enc_feat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(hidden * 2, hidden, kernel_size=4, stride=2, padding=1),  # x2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden, in_channels, kernel_size=4, stride=2, padding=1),  # x2
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h4 = self.patch_size // 4
        h = self.fc_z(z).view(z.size(0), -1, h4, h4)
        x_hat = self.dec(h)
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def vae_loss(x, x_hat, mu, logvar, beta: float = 1.0):
    """
    Reconstruction + beta * KL
    """
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kl = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return recon + beta * kl, recon.detach(), kl.detach()
