"""
train_branch_b.py — Standalone training script for Speaker Verification.

Usage
-----
    python src/train_branch_b.py --config config/branch_b_config.yaml
    python src/train_branch_b.py --config config/branch_b_config.yaml --loss contrastive

Maps to ATML Unit 1: Siamese networks + few-shot / meta-learning.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.voxceleb_dataset import VoxCelebDataset
from src.models.branch_b_siamese import SiameseSpeakerVerifier
from src.losses.triplet_loss import TripletLoss
from src.losses.contrastive_loss import ContrastiveLoss
from src.datasets.audio_utils import load_and_preprocess

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
    req_device = cfg.get("device", "cuda")
    if req_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    logger.info("Using CPU device.")
    return torch.device("cpu")


def compute_eer_from_scores(scores: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Compute EER and optimal threshold from cosine similarity scores (higher = more similar).
    labels: 1 = same speaker, 0 = different.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thresholds[idx])


def compute_eer_from_pairs(
    model: SiameseSpeakerVerifier,
    pairs: list,
    audio_cfg: dict,
    device: torch.device,
) -> tuple:
    if not pairs:
        return 0.5, 0.0

    model.eval()
    scores = []
    labels = []

    with torch.no_grad():
        for path_a, path_b, is_same in pairs:
            if not os.path.exists(path_a) or not os.path.exists(path_b):
                continue
            mel_a = load_and_preprocess(path_a, audio_cfg, augment=False).unsqueeze(0).to(device)
            mel_b = load_and_preprocess(path_b, audio_cfg, augment=False).unsqueeze(0).to(device)

            emb_a = model.encode(mel_a)
            emb_b = model.encode(mel_b)
            sim = model.compute_similarity(emb_a, emb_b).item()

            scores.append(sim)
            labels.append(is_same)

    if len(scores) < 2 or len(set(labels)) < 2:
        return 0.5, 0.0

    return compute_eer_from_scores(np.array(scores), np.array(labels))


def train_one_epoch(
    model: SiameseSpeakerVerifier,
    dataloader: DataLoader,
    loss_fn,
    loss_type: str,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_clip: float = 5.0,
) -> float:
    model.train()
    total_loss = 0.0
    count = 0

    for batch in dataloader:
        optimizer.zero_grad()

        if loss_type == "triplet":
            anchor_mel, pos_mel, neg_mel, labels = batch
            anchor_mel = anchor_mel.to(device)
            pos_mel = pos_mel.to(device)
            neg_mel = neg_mel.to(device)
            labels = labels.to(device)

            out = model.forward(anchor_mel, pos_mel, neg_mel)
            # Stack anchor, positive, and negative embeddings along with labels for online mining
            all_embs = torch.cat([out["anchor_emb"], out["positive_emb"], out["negative_emb"]], dim=0)
            # Create corresponding speaker IDs
            # anchor and positive have the same label, negative has offset label
            neg_labels = labels + 100000
            all_labels = torch.cat([labels, labels, neg_labels], dim=0)

            loss = loss_fn(all_embs, all_labels)

        else:  # contrastive
            mel_a, mel_b, is_same = batch
            mel_a = mel_a.to(device)
            mel_b = mel_b.to(device)
            is_same = is_same.to(device)

            emb_a = model.encode(mel_a)
            emb_b = model.encode(mel_b)
            loss = loss_fn(emb_a, emb_b, is_same)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        count += 1

    return total_loss / max(1, count)


def main(args: argparse.Namespace) -> None:
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.loss:
        cfg["loss"]["type"] = args.loss

    set_seeds(cfg.get("seed", 42))
    device = get_device(cfg)

    audio_cfg = cfg["audio"]
    loss_type = cfg["loss"]["type"]

    logger.info("Initializing VoxCeleb dataset (mode: %s)...", loss_type)
    train_ds = VoxCelebDataset(
        voxceleb_root=cfg["data"]["voxceleb1_root"],
        audio_cfg=audio_cfg,
        mode=loss_type,
        split="train",
        num_speakers=cfg["data"].get("num_train_speakers", 300),
        augment=True,
    )

    if len(train_ds) == 0:
        logger.warning("VoxCeleb dataset empty or directory not found. Please run data/download_voxceleb.sh.")

    train_loader = DataLoader(
        train_ds,
        batch_size=min(cfg["training"]["batch_size"], max(1, len(train_ds))),
        shuffle=True,
        drop_last=len(train_ds) > cfg["training"]["batch_size"],
    )

    logger.info("Instantiating Siamese speaker verifier (encoder: %s)...", cfg["encoder"]["type"])
    model = SiameseSpeakerVerifier(cfg).to(device)

    # Initialize loss
    if loss_type == "triplet":
        loss_fn = TripletLoss(
            margin=cfg["loss"].get("margin", 0.2),
            mining=cfg["loss"].get("mining", "semihard"),
            distance="cosine",
        )
    elif loss_type == "contrastive":
        loss_fn = ContrastiveLoss(
            margin=cfg["loss"].get("contrastive_margin", 1.0),
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )

    veri_pairs = train_ds.get_verification_pairs(cfg["data"]["veri_test_pairs"])
    logger.info("Found %d official verification pairs for evaluation.", len(veri_pairs))

    checkpoint_dir = Path(cfg["checkpoints"]["dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_eer = float("inf")
    best_threshold = 0.5
    epochs = cfg["training"]["epochs"]

    for epoch in range(1, epochs + 1):
        if len(train_ds) > 0:
            avg_loss = train_one_epoch(
                model=model,
                dataloader=train_loader,
                loss_fn=loss_fn,
                loss_type=loss_type,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                grad_clip=cfg["training"].get("gradient_clip_norm", 5.0),
            )
            logger.info("Epoch %02d/%02d - Train Loss: %.4f", epoch, epochs, avg_loss)

        if veri_pairs:
            eer, threshold = compute_eer_from_pairs(model, veri_pairs, audio_cfg, device)
            logger.info("Epoch %02d/%02d - Val EER: %.2f%% (Threshold: %.3f)", epoch, epochs, eer * 100.0, threshold)

            if eer < best_eer:
                best_eer = eer
                best_threshold = threshold
                best_path = checkpoint_dir / "best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_eer": best_eer,
                    "threshold": best_threshold,
                    "config": cfg,
                }, best_path)
                logger.info("Saved new best model to %s", best_path)

    # Save final eer_threshold.json
    thresh_file = checkpoint_dir / "eer_threshold.json"
    with open(thresh_file, "w", encoding="utf-8") as f:
        json.dump({"eer_threshold": best_threshold, "best_val_eer": best_eer}, f, indent=2)
    logger.info("EER threshold saved to %s", thresh_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Branch B — Speaker Verification")
    parser.add_argument("--config", required=True)
    parser.add_argument("--loss", choices=["triplet", "contrastive"], default=None)
    args = parser.parse_args()
    main(args)
