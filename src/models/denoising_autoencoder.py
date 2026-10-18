"""
denoising_autoencoder.py — Mel-spectrogram Denoising Autoencoder.

Maps to ATML Unit 2: Autoencoder-based anomaly / outlier detection.

Purpose
-------
Trained ONLY on bonafide (genuine human speech) log-Mel spectrograms.
At inference time, candidate speech clips are passed through the autoencoder.
Reconstruction error (MSE) is computed. Genuine speech reconstructs accurately
with low error; synthetic speech (TTS, VC, neural vocoder) exhibits higher
reconstruction error due to unseen spectral anomalies.

The reconstruction error is normalized (z-score) and fed as an independent
input to the Fusion layer.
"""

import json
import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SpectrogramEncoder(nn.Module):
    """
    2-D Convolutional encoder over log-Mel spectrograms.
    Input: [B, n_mels, T_frames] -> Latent: [B, latent_dim]
    """

    def __init__(
        self,
        n_mels: int = 80,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B, 32, n_mels//2, T//2]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B, 64, n_mels//4, T//4]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B, 128, n_mels//8, T//8]
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, n_mels, T]
        """
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = x.unsqueeze(1)  # [B, 1, n_mels, T]
        features = self.conv(x)
        pooled = self.pool(features)
        flat = torch.flatten(pooled, start_dim=1)
        z = self.fc(flat)
        return z


class SpectrogramDecoder(nn.Module):
    """
    Decodes latent vector z [B, latent_dim] back to reconstructed spectrogram [B, n_mels, T_frames].
    """

    def __init__(
        self,
        latent_dim: int = 128,
        out_mels: int = 80,
    ) -> None:
        super().__init__()
        self.out_mels = out_mels
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, z: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
        """
        z: [B, latent_dim]
        target_shape: (n_mels, T_frames)
        """
        b = z.size(0)
        x = self.fc(z).view(b, 128, 4, 4)
        x = self.deconv(x)  # [B, 1, H', W']
        # Rescale exactly to target_shape (frequency, time)
        reconstruction = F.interpolate(x, size=target_shape, mode="bilinear", align_corners=False)
        return reconstruction.squeeze(1)


class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder for spectrogram anomaly detection.
    """

    def __init__(
        self,
        n_mels: int = 80,
        latent_dim: int = 128,
        noise_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.latent_dim = latent_dim
        self.noise_std = noise_std

        self.encoder = SpectrogramEncoder(n_mels=n_mels, latent_dim=latent_dim)
        self.decoder = SpectrogramDecoder(latent_dim=latent_dim, out_mels=n_mels)

    def forward(
        self,
        x: torch.Tensor,
        add_noise: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        x: [B, n_mels, T]
        """
        if x.ndim == 2:
            x = x.unsqueeze(0)

        target_shape = (x.size(-2), x.size(-1))

        if self.training and add_noise and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x_in = x + noise
        else:
            x_in = x

        z = self.encoder(x_in)
        reconstruction = self.decoder(z, target_shape)
        loss = F.mse_loss(reconstruction, x)

        return {
            "reconstruction": reconstruction,
            "loss": loss,
            "latent": z,
        }

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute sample-wise MSE reconstruction error.
        x: [B, n_mels, T]
        returns: [B] per-sample MSE error
        """
        self.eval()
        if x.ndim == 2:
            x = x.unsqueeze(0)
        with torch.no_grad():
            out = self.forward(x, add_noise=False)
            reconstruction = out["reconstruction"]
            mse_per_sample = torch.mean((x - reconstruction) ** 2, dim=(-2, -1))
        return mse_per_sample


# ---------------------------------------------------------------------------
# Normalization Statistics Utilities
# ---------------------------------------------------------------------------

def compute_normalisation_stats(
    autoencoder: DenoisingAutoencoder,
    bonafide_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute mean and std of reconstruction errors on bonafide speech.
    """
    autoencoder.eval()
    errors = []

    with torch.no_grad():
        for batch in bonafide_dataloader:
            feature = batch["feature"].to(device)
            err = autoencoder.reconstruction_error(feature)
            errors.extend(err.cpu().numpy().tolist())

    if len(errors) == 0:
        return {"mean": 0.1, "std": 0.05, "min": 0.0, "max": 0.5}

    arr = np.array(errors)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr) + 1e-8),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def save_normalisation_stats(stats: Dict[str, float], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def load_normalisation_stats(path: str) -> Dict[str, float]:
    if not os.path.exists(path):
        return {"mean": 0.1, "std": 0.05, "min": 0.0, "max": 0.5}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalise_reconstruction_error(
    error: float,
    stats: Dict[str, float],
    method: str = "z_score",
) -> float:
    """
    Normalize raw MSE error using bonafide validation stats.
    """
    mean = stats.get("mean", 0.0)
    std = stats.get("std", 1.0)

    if method == "z_score":
        return float((error - mean) / (std + 1e-8))
    elif method == "minmax":
        min_v = stats.get("min", 0.0)
        max_v = stats.get("max", 1.0)
        return float(np.clip((error - min_v) / (max_v - min_v + 1e-8), 0.0, 1.0))
    else:
        return float(error)
