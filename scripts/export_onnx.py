"""
export_onnx.py — Export trained models to ONNX for lightweight edge deployment.

Usage
-----
    python scripts/export_onnx.py --config config/fusion_config.yaml --model branch_a
    python scripts/export_onnx.py --config config/fusion_config.yaml --model branch_b
    python scripts/export_onnx.py --config config/fusion_config.yaml --model autoencoder
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.branch_a_spoof_detector import SpoofDetector
from src.models.branch_b_siamese import SiameseSpeakerVerifier
from src.models.denoising_autoencoder import DenoisingAutoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def export_branch_a(cfg: dict, device: torch.device) -> str:
    with open(cfg["branch_a"]["config"], "r", encoding="utf-8") as f:
        branch_a_cfg = yaml.safe_load(f)

    # Use cnn_attention mode for reliable ONNX export without huggingface transformer opset issues
    branch_a_cfg_copy = dict(branch_a_cfg)
    branch_a_cfg_copy["backbone"] = {"name": "cnn_attention", "output_dim": 256}

    model = SpoofDetector(branch_a_cfg_copy).to(device)
    model.eval()

    n_mels = branch_a_cfg_copy.get("audio", {}).get("n_mels", 80)
    dummy_input = torch.randn(1, n_mels, 200, device=device)

    out_dir = Path("checkpoints/branch_a")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "spoof_detector.onnx"

    class ExportWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x)["probs"]

    wrapper = ExportWrapper(model)
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(out_path),
        input_names=["mel_spectrogram"],
        output_names=["probabilities"],
        dynamic_axes={"mel_spectrogram": {0: "batch_size", 2: "time_frames"}, "probabilities": {0: "batch_size"}},
        opset_version=14,
    )
    logger.info("Exported Branch A ONNX to %s", out_path)
    return str(out_path)


def export_branch_b(cfg: dict, device: torch.device) -> str:
    with open(cfg["branch_b"]["config"], "r", encoding="utf-8") as f:
        branch_b_cfg = yaml.safe_load(f)

    model = SiameseSpeakerVerifier(branch_b_cfg).to(device)
    model.eval()

    n_mels = branch_b_cfg.get("audio", {}).get("n_mels", 80)
    dummy_input = torch.randn(1, n_mels, 200, device=device)

    out_dir = Path("checkpoints/branch_b")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "speaker_encoder.onnx"

    class ExportWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m.encode(x)

    wrapper = ExportWrapper(model)
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(out_path),
        input_names=["mel_spectrogram"],
        output_names=["embedding"],
        dynamic_axes={"mel_spectrogram": {0: "batch_size", 2: "time_frames"}, "embedding": {0: "batch_size"}},
        opset_version=14,
    )
    logger.info("Exported Branch B ONNX to %s", out_path)
    return str(out_path)


def export_autoencoder(cfg: dict, device: torch.device) -> str:
    with open(cfg["branch_a"]["config"], "r", encoding="utf-8") as f:
        branch_a_cfg = yaml.safe_load(f)

    n_mels = branch_a_cfg.get("audio", {}).get("n_mels", 80)
    model = DenoisingAutoencoder(n_mels=n_mels, latent_dim=128).to(device)
    model.eval()

    dummy_input = torch.randn(1, n_mels, 200, device=device)
    out_dir = Path("checkpoints/autoencoder")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "autoencoder.onnx"

    class ExportWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x, add_noise=False)["reconstruction"]

    wrapper = ExportWrapper(model)
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(out_path),
        input_names=["mel_spectrogram"],
        output_names=["reconstruction"],
        dynamic_axes={"mel_spectrogram": {0: "batch_size", 2: "time_frames"}, "reconstruction": {0: "batch_size", 2: "time_frames"}},
        opset_version=14,
    )
    logger.info("Exported Autoencoder ONNX to %s", out_path)
    return str(out_path)


def main(args: argparse.Namespace) -> None:
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")

    if args.model == "branch_a":
        export_branch_a(cfg, device)
    elif args.model == "branch_b":
        export_branch_b(cfg, device)
    elif args.model == "autoencoder":
        export_autoencoder(cfg, device)
    elif args.model == "all":
        export_branch_a(cfg, device)
        export_branch_b(cfg, device)
        export_autoencoder(cfg, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch models to ONNX")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=["branch_a", "branch_b", "autoencoder", "all"], default="all")
    args = parser.parse_args()
    main(args)
