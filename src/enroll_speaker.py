"""
enroll_speaker.py — CLI for few-shot speaker enrollment.

Usage
-----
    # Enroll a new speaker from 3-5 reference clips
    python src/enroll_speaker.py \
        --config config/fusion_config.yaml \
        --speaker-id "alice" \
        --clips path/to/clip1.wav path/to/clip2.wav path/to/clip3.wav

    # List enrolled speakers
    python src/enroll_speaker.py --config config/fusion_config.yaml --list

    # Delete an enrolled speaker
    python src/enroll_speaker.py --config config/fusion_config.yaml --delete "alice"

Maps to ATML Unit 1: Few-shot / meta-learning via average embedding aggregation.
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.audio_utils import load_and_preprocess
from src.models.branch_b_siamese import SiameseSpeakerVerifier, enroll_speaker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_enrollment_store(path: str) -> Dict[str, torch.Tensor]:
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning("Could not read enrollment store from %s: %s", path, e)
    return {}


def save_enrollment_store(store: Dict[str, torch.Tensor], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(store, f)


def main(args: argparse.Namespace) -> None:
    with open(args.config, "r", encoding="utf-8") as f:
        fusion_cfg = yaml.safe_load(f)

    branch_b_cfg_path = fusion_cfg["branch_b"]["config"]
    with open(branch_b_cfg_path, "r", encoding="utf-8") as f:
        branch_b_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if fusion_cfg.get("device") == "cuda" and torch.cuda.is_available() else "cpu")
    enrollment_store_path = fusion_cfg["branch_b"].get("enrollment_store", "checkpoints/branch_b/enrollments.pkl")
    store = load_enrollment_store(enrollment_store_path)

    if args.list:
        print("\n=== Enrolled Speakers ===")
        if not store:
            print("  (No speakers enrolled yet)")
        else:
            for spk_id, emb in store.items():
                print(f"  - {spk_id:20s} (Embedding shape: {list(emb.shape)})")
        print("=========================\n")
        return

    if args.delete:
        if args.delete in store:
            del store[args.delete]
            save_enrollment_store(store, enrollment_store_path)
            logger.info("Deleted speaker '%s' from enrollment store.", args.delete)
        else:
            logger.warning("Speaker '%s' not found in store.", args.delete)
        return

    if args.speaker_id and args.clips:
        n_clips = len(args.clips)
        v_min = branch_b_cfg.get("verification", {}).get("enrollment_clips_min", 3)
        v_max = branch_b_cfg.get("verification", {}).get("enrollment_clips_max", 5)

        if not (v_min <= n_clips <= v_max):
            logger.warning("Recommended enrollment is between %d and %d clips (received %d). Proceeding...", v_min, v_max, n_clips)

        # Initialize Siamese model
        model = SiameseSpeakerVerifier(branch_b_cfg).to(device)
        ckpt_path = fusion_cfg["branch_b"].get("checkpoint")
        if ckpt_path and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            logger.info("Loaded speaker verifier weights from %s", ckpt_path)
        else:
            logger.info("No checkpoint found at %s. Using initialized encoder.", ckpt_path)

        audio_cfg = branch_b_cfg["audio"]
        mel_specs = []
        for clip in args.clips:
            if not os.path.exists(clip):
                logger.error("Audio clip file not found: %s", clip)
                sys.exit(1)
            mel = load_and_preprocess(clip, audio_cfg, augment=False, random_crop=False)
            mel_specs.append(mel)

        avg_emb = enroll_speaker(model, mel_specs, device)
        store[args.speaker_id] = avg_emb.cpu()
        save_enrollment_store(store, enrollment_store_path)
        logger.info("Successfully enrolled speaker '%s' using %d reference clips.", args.speaker_id, len(mel_specs))
        return

    logger.error("Please supply --list, --delete <id>, or --speaker-id <id> with --clips <paths...>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll a speaker (few-shot)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--speaker-id", default=None)
    parser.add_argument("--clips", nargs="+", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--delete", default=None)
    args = parser.parse_args()
    main(args)
