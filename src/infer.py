"""
infer.py — End-to-end inference: audio in -> risk score + verdict out.

Usage
-----
    # Without speaker verification (spoof detection + anomaly detection)
    python src/infer.py --config config/fusion_config.yaml --audio sample.wav

    # With speaker verification (enrolled speaker required)
    python src/infer.py \
        --config config/fusion_config.yaml \
        --audio sample.wav \
        --speaker-id "alice"
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.audio_utils import load_and_preprocess, load_waveform, pad_or_trim
from src.models.branch_a_spoof_detector import SpoofDetector
from src.models.branch_b_siamese import SiameseSpeakerVerifier
from src.models.denoising_autoencoder import (
    DenoisingAutoencoder,
    load_normalisation_stats,
    normalise_reconstruction_error,
)
from src.models.fusion import FusionEngine
from src.enroll_speaker import load_enrollment_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_all_models(
    fusion_cfg: dict,
    device: torch.device,
) -> Tuple[SpoofDetector, SiameseSpeakerVerifier, DenoisingAutoencoder, dict, FusionEngine, dict, dict]:
    # Branch A
    with open(fusion_cfg["branch_a"]["config"], "r", encoding="utf-8") as f:
        branch_a_cfg = yaml.safe_load(f)
    ckpt_a = fusion_cfg["branch_a"].get("checkpoint")
    st_a = None
    if ckpt_a and os.path.exists(ckpt_a):
        st_a = torch.load(ckpt_a, map_location=device)
        if "config" in st_a and isinstance(st_a["config"], dict):
            # Inherit trained architecture settings from checkpoint
            for section in ("backbone", "attention_pooling", "head"):
                if section in st_a["config"]:
                    branch_a_cfg[section] = st_a["config"][section]
        elif "model_state" in st_a and "head.0.weight" in st_a["model_state"]:
            # Auto-detect hidden dimension from weight shape
            trained_hidden_dim = st_a["model_state"]["head.0.weight"].shape[0]
            branch_a_cfg["head"]["hidden_dim"] = trained_hidden_dim

    model_a = SpoofDetector(branch_a_cfg).to(device)
    if st_a is not None:
        model_a.load_state_dict(st_a["model_state"], strict=False)
        logger.info("Loaded Branch A checkpoint: %s", ckpt_a)
    model_a.eval()

    # Branch B
    with open(fusion_cfg["branch_b"]["config"], "r", encoding="utf-8") as f:
        branch_b_cfg = yaml.safe_load(f)
    ckpt_b = fusion_cfg["branch_b"].get("checkpoint")
    st_b = None
    if ckpt_b and os.path.exists(ckpt_b):
        st_b = torch.load(ckpt_b, map_location=device)
        if "config" in st_b and isinstance(st_b["config"], dict):
            for section in ("encoder", "audio"):
                if section in st_b["config"]:
                    branch_b_cfg[section] = st_b["config"][section]

    model_b = SiameseSpeakerVerifier(branch_b_cfg).to(device)
    if st_b is not None:
        model_b.load_state_dict(st_b["model_state"], strict=False)
        logger.info("Loaded Branch B checkpoint: %s", ckpt_b)
    model_b.eval()

    # Autoencoder
    ae = DenoisingAutoencoder(
        n_mels=branch_a_cfg.get("audio", {}).get("n_mels", 80),
        latent_dim=128,
    ).to(device)
    ckpt_ae = fusion_cfg["autoencoder"].get("checkpoint")
    if ckpt_ae and os.path.exists(ckpt_ae):
        st = torch.load(ckpt_ae, map_location=device)
        ae.load_state_dict(st["model_state"])
        logger.info("Loaded Autoencoder checkpoint: %s", ckpt_ae)
    ae.eval()

    stats_file = Path(ckpt_ae).parent / "autoencoder_stats.json" if ckpt_ae else None
    ae_stats = load_normalisation_stats(str(stats_file)) if stats_file else {"mean": 0.1, "std": 0.05}

    fusion_engine = FusionEngine.from_config(fusion_cfg)

    return model_a, model_b, ae, ae_stats, fusion_engine, branch_a_cfg, branch_b_cfg


@torch.no_grad()
def run_inference(
    audio_path: str,
    fusion_cfg: dict,
    device: torch.device,
    speaker_id: Optional[str] = None,
    preloaded_models: Optional[tuple] = None,
) -> Dict:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if preloaded_models is not None:
        model_a, model_b, ae, ae_stats, fusion_engine, branch_a_cfg, branch_b_cfg = preloaded_models
    else:
        model_a, model_b, ae, ae_stats, fusion_engine, branch_a_cfg, branch_b_cfg = load_all_models(fusion_cfg, device)

    audio_cfg_a = branch_a_cfg["audio"]
    audio_cfg_b = branch_b_cfg["audio"]

    # 1. Branch A Inference
    backbone_name = branch_a_cfg["backbone"].get("name", "wav2vec2")
    if backbone_name == "wav2vec2":
        wav, _ = load_waveform(audio_path, target_sr=audio_cfg_a.get("sample_rate", 16000), mono=True)
        wav = pad_or_trim(wav, max_duration_s=audio_cfg_a.get("max_duration_s", 4.0), sample_rate=audio_cfg_a.get("sample_rate", 16000))
        input_a = wav.squeeze(0).unsqueeze(0).to(device)  # [1, T]
    else:
        mel_a = load_and_preprocess(audio_path, audio_cfg_a, augment=False, random_crop=False)
        input_a = mel_a.unsqueeze(0).to(device)  # [1, n_mels, T]

    out_a = model_a.forward(input_a)
    spoof_prob = float(out_a["probs"][0, 1].item())

    # 2. Autoencoder Anomaly Signal Inference
    mel_ae = load_and_preprocess(audio_path, audio_cfg_a, augment=False, random_crop=False)
    raw_error = float(ae.reconstruction_error(mel_ae.unsqueeze(0).to(device))[0].item())
    norm_method = fusion_cfg.get("autoencoder", {}).get("reconstruction_normalization", "z_score")
    ae_error_norm = normalise_reconstruction_error(raw_error, ae_stats, method=norm_method)

    # 3. Branch B Speaker Verification Inference
    speaker_sim = 0.0  # neutral match if unverified
    if speaker_id is not None:
        store_path = fusion_cfg["branch_b"].get("enrollment_store", "checkpoints/branch_b/enrollments.pkl")
        store = load_enrollment_store(store_path)

        if speaker_id in store:
            enrolled_emb = store[speaker_id].to(device)
            mel_b = load_and_preprocess(audio_path, audio_cfg_b, augment=False, random_crop=False)
            query_emb = model_b.encode(mel_b.unsqueeze(0).to(device))
            speaker_sim = float(model_b.compute_similarity(query_emb, enrolled_emb).item())
        else:
            logger.warning("Speaker '%s' not found in enrollment store. Defaulting similarity to 0.0.", speaker_id)

    # 4. Multimodal Fusion
    result = fusion_engine.score(
        spoof_prob=spoof_prob,
        ae_error_norm=ae_error_norm,
        speaker_similarity=speaker_sim,
    )
    result["raw_reconstruction_error"] = raw_error
    result["claimed_speaker_id"] = speaker_id

    return result


def main(args: argparse.Namespace) -> None:
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if cfg.get("device") == "cuda" and torch.cuda.is_available() else "cpu")

    result = run_inference(
        audio_path=args.audio,
        fusion_cfg=cfg,
        device=device,
        speaker_id=args.speaker_id,
    )

    print("\n" + "=" * 60)
    print(f"  VOICE CLONE GUARD — RISK ASSESSMENT")
    print("=" * 60)
    print(f"  FINAL RISK SCORE   : {result['risk_score']:.1f} / 100")
    print(f"  VERDICT            : {result['verdict']}")
    print("-" * 60)
    print(f"  Branch A (P_spoof) : {result['spoof_probability']:.4f}")
    print(f"  Autoencoder Error  : {result['ae_error_normalised']:.4f} (z-score)")
    print(f"  Branch B (Match)   : {result['speaker_similarity']:.4f} (claimed: {result['claimed_speaker_id']})")
    print("-" * 60)
    print("  Diagnostic Attribution:")
    for k, v in result["explanation"].items():
        if "percentage" in k:
            print(f"    - {k:22s}: {v}%")
    print("-" * 60)
    print(f"  NOTE: {result['disclaimer']}")
    print("=" * 60 + "\n")

    if args.json:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end voice cloning & speaker verification inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--speaker-id", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    main(args)
