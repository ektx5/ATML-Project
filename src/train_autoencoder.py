"""
train_autoencoder.py — Standalone training script for Denoising Autoencoder.

Maps to ATML Unit 2: Autoencoder-based anomaly / outlier detection.

Usage
-----
    python src/train_autoencoder.py --config config/fusion_config.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.asvspoof_dataset import ASVspoofDataset
from src.models.denoising_autoencoder import (
    DenoisingAutoencoder,
    compute_normalisation_stats,
    save_normalisation_stats,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def filter_bonafide(dataset: ASVspoofDataset) -> Subset:
    """
    Return a PyTorch Subset with only bonafide samples (label_int == 0).
    """
    indices = [i for i, row in dataset.df.iterrows() if int(row["label_int"]) == 0]
    return Subset(dataset, indices)


def train_one_epoch(
    model: DenoisingAutoencoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        features = batch["feature"].to(device)
        optimizer.zero_grad()
        out = model.forward(features, add_noise=True)
        loss = out["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def validate(
    model: DenoisingAutoencoder,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0

    for batch in dataloader:
        features = batch["feature"].to(device)
        out = model.forward(features, add_noise=False)
        total_loss += out["loss"].item()

    return total_loss / max(1, len(dataloader))


def main(args: argparse.Namespace) -> None:
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load audio configuration from branch_a_config.yaml
    with open(cfg["branch_a"]["config"], "r", encoding="utf-8") as f:
        branch_a_cfg = yaml.safe_load(f)

    audio_cfg = branch_a_cfg["audio"]
    device = torch.device("cuda" if cfg.get("device") == "cuda" and torch.cuda.is_available() else "cpu")

    logger.info("Loading ASVspoof dataset to extract bonafide utterances...")
    train_ds = ASVspoofDataset(
        audio_dir=branch_a_cfg["data"]["train_dir"],
        protocol_path=branch_a_cfg["data"]["train_protocol"],
        audio_cfg=audio_cfg,
        split="train",
        augment=False,
        return_waveform=False,  # autoencoder operates on Mel spectrograms
    )
    dev_ds = ASVspoofDataset(
        audio_dir=branch_a_cfg["data"]["dev_dir"],
        protocol_path=branch_a_cfg["data"]["dev_protocol"],
        audio_cfg=audio_cfg,
        split="dev",
        augment=False,
        return_waveform=False,
    )

    train_bonafide = filter_bonafide(train_ds)
    dev_bonafide = filter_bonafide(dev_ds)

    logger.info("Bonafide subsets: Train=%d, Dev=%d", len(train_bonafide), len(dev_bonafide))

    batch_size = 32
    train_loader = DataLoader(
        train_bonafide,
        batch_size=min(batch_size, max(1, len(train_bonafide))),
        shuffle=True,
    )
    dev_loader = DataLoader(
        dev_bonafide,
        batch_size=min(batch_size, max(1, len(dev_bonafide))),
        shuffle=False,
    )

    model = DenoisingAutoencoder(
        n_mels=audio_cfg.get("n_mels", 80),
        latent_dim=128,
        noise_std=0.05,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    ckpt_path = Path(cfg["autoencoder"]["checkpoint"])
    ckpt_dir = ckpt_path.parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    epochs = 20

    for epoch in range(1, epochs + 1):
        if len(train_bonafide) > 0:
            train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
            logger.info("Epoch %02d/%02d - Autoencoder Train MSE: %.6f", epoch, epochs, train_loss)

        if len(dev_bonafide) > 0:
            val_loss = validate(model, dev_loader, device)
            logger.info("Epoch %02d/%02d - Autoencoder Val MSE: %.6f", epoch, epochs, val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({"model_state": model.state_dict(), "val_loss": val_loss}, ckpt_path)
                logger.info("Saved best autoencoder to %s", ckpt_path)

    # Compute and save normalisation statistics from bonafide dev set
    stats = compute_normalisation_stats(model, dev_loader, device)
    stats_file = ckpt_dir / "autoencoder_stats.json"
    save_normalisation_stats(stats, str(stats_file))
    logger.info("Saved autoencoder normalisation stats to %s: %s", stats_file, stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Denoising Autoencoder for Anomaly Detection")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args)
