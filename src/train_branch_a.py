"""
train_branch_a.py — Standalone training script for Branch A (Spoof Detector).

Usage
-----
    python src/train_branch_a.py --config config/branch_a_config.yaml
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.asvspoof_dataset import ASVspoofDataset
from src.models.branch_a_spoof_detector import SpoofDetector, save_checkpoint, load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_device(cfg: dict) -> torch.device:
    req = cfg.get("device", "cuda")
    if req == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    logger.info("Using CPU device.")
    return torch.device("cpu")


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple:
    """
    scores: P(spoof) in [0, 1]
    labels: 0 = bonafide, 1 = spoof
    """
    if len(scores) < 2 or len(set(labels)) < 2:
        return 0.5, 0.5
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thresholds[idx])


def train_one_epoch(
    model: SpoofDetector,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_clip: float = 5.0,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        features = batch["feature"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        out = model.forward(features, labels=labels)
        loss = out["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        preds = torch.argmax(out["probs"], dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()

    avg_loss = total_loss / max(1, len(dataloader))
    acc = correct / max(1, total)
    return avg_loss, acc


@torch.no_grad()
def validate(
    model: SpoofDetector,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    all_scores = []
    all_labels = []

    for batch in dataloader:
        features = batch["feature"].to(device)
        labels = batch["label"].to(device)

        out = model.forward(features, labels=labels)
        total_loss += out["loss"].item()

        spoof_probs = out["probs"][:, 1].cpu().numpy()
        all_scores.extend(spoof_probs)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(1, len(dataloader))
    if len(all_scores) > 0 and len(set(all_labels)) > 1:
        eer, _ = compute_eer(np.array(all_scores), np.array(all_labels))
        preds = (np.array(all_scores) >= 0.5).astype(int)
        acc = (preds == np.array(all_labels)).mean()
    else:
        eer = 0.5
        acc = 0.5

    return avg_loss, acc, eer


def main(args: argparse.Namespace) -> None:
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seeds(cfg.get("seed", 42))
    device = get_device(cfg)

    backbone_name = cfg["backbone"].get("name", "wav2vec2")
    return_wav = (backbone_name == "wav2vec2")

    logger.info("Initializing ASVspoof datasets...")
    train_ds = ASVspoofDataset(
        audio_dir=cfg["data"]["train_dir"],
        protocol_path=cfg["data"]["train_protocol"],
        audio_cfg=cfg["audio"],
        split="train",
        augment=True,
        return_waveform=return_wav,
    )
    dev_ds = ASVspoofDataset(
        audio_dir=cfg["data"]["dev_dir"],
        protocol_path=cfg["data"]["dev_protocol"],
        audio_cfg=cfg["audio"],
        split="dev",
        augment=False,
        return_waveform=return_wav,
    )

    if len(train_ds) == 0:
        logger.warning("Train protocol/audio not found. Please run data/download_asvspoof.sh.")

    # Class weighting
    class_weights = None
    if cfg["training"].get("use_class_weights", True) and len(train_ds) > 0:
        class_weights = train_ds.class_weights.to(device)
        logger.info("Computed class weights: Bonafide=%.3f, Spoof=%.3f", class_weights[0].item(), class_weights[1].item())

    sampler = None
    if cfg["training"].get("balanced_sampler", False) and len(train_ds) > 0:
        sampler = train_ds.get_weighted_sampler()

    train_loader = DataLoader(
        train_ds,
        batch_size=min(cfg["training"]["batch_size"], max(1, len(train_ds))),
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=len(train_ds) > cfg["training"]["batch_size"],
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=min(cfg["training"]["batch_size"], max(1, len(dev_ds))),
        shuffle=False,
    )

    logger.info("Instantiating SpoofDetector (Backbone: %s)...", backbone_name)
    model = SpoofDetector(cfg, class_weights=class_weights).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )

    if args.resume and os.path.exists(args.resume):
        load_checkpoint(model, args.resume, device, optimizer)
        logger.info("Resumed checkpoint from %s", args.resume)

    checkpoint_dir = Path(cfg["checkpoints"]["dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_eer = float("inf")
    epochs = cfg["training"]["epochs"]

    for epoch in range(1, epochs + 1):
        if len(train_ds) > 0:
            train_loss, train_acc = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                grad_clip=cfg["training"].get("gradient_clip_norm", 5.0),
            )
            logger.info("Epoch %02d/%02d - Train Loss: %.4f | Train Acc: %.2f%%", epoch, epochs, train_loss, train_acc * 100.0)

        if len(dev_ds) > 0:
            val_loss, val_acc, val_eer = validate(model, dev_loader, device)
            logger.info("Epoch %02d/%02d - Val Loss: %.4f | Val Acc: %.2f%% | Val EER: %.2f%%", epoch, epochs, val_loss, val_acc * 100.0, val_eer * 100.0)

            if val_eer < best_val_eer:
                best_val_eer = val_eer
                best_path = str(checkpoint_dir / "best.pt")
                save_checkpoint(model, optimizer, epoch, val_eer, best_path)
                logger.info("Saved new best model to %s", best_path)

    logger.info("Branch A training finished. Best Val EER: %.2f%%", best_val_eer * 100.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Branch A — Spoof Detector")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    main(args)
