#!/usr/bin/env bash
# run_full_pipeline.sh — One-command end-to-end pipeline.
#
# Usage:
#   bash scripts/run_full_pipeline.sh
#
# Steps:
#   1. Verify datasets are present (does not download — that requires manual registration)
#   2. Preprocess: verify audio configs are consistent
#   3. Train autoencoder (ATML Unit 2)
#   4. Train Branch A — spoof detector (ATML Unit 3+4)
#   5. Train Branch B — speaker verifier (ATML Unit 1)
#   6. Evaluate: ASVspoof eval + In-the-Wild + VoxCeleb verification pairs
#   7. Run ablation study
#
# Prerequisites:
#   1. Activate conda environment: conda activate voice-clone-guard
#   2. Download datasets (see data/README.md)
#   3. Set WANDB_API_KEY if using Weights & Biases

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "========================================================"
echo "  Voice Clone Guard — Full Training Pipeline"
echo "========================================================"
echo ""

# ---------- 0. Verify environment ----------
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}')"

# ---------- 1. Verify datasets ----------
echo ""
echo "[1/6] Verifying dataset directories..."
for DIR in "datasets/ASVspoof2019_LA" "datasets/VoxCeleb1"; do
    if [ ! -d "$DIR" ]; then
        echo "[ERROR] Dataset directory not found: $DIR"
        echo "        See data/README.md for download instructions."
        exit 1
    fi
done
echo "     Datasets OK."

# ---------- 2. Train autoencoder ----------
echo ""
echo "[2/6] Training denoising autoencoder (ATML Unit 2)..."
python src/train_autoencoder.py --config config/fusion_config.yaml

# ---------- 3. Train Branch A ----------
echo ""
echo "[3/6] Training Branch A — Spoof Detector (ATML Unit 3+4)..."
python src/train_branch_a.py --config config/branch_a_config.yaml

# ---------- 4. Train Branch B ----------
echo ""
echo "[4/6] Training Branch B — Speaker Verifier (ATML Unit 1)..."
python src/train_branch_b.py --config config/branch_b_config.yaml

# ---------- 5+6. Evaluate (all splits + ablation) ----------
echo ""
echo "[5/6] Running full evaluation..."
python src/evaluate.py --config config/fusion_config.yaml --split all

echo ""
echo "========================================================"
echo "  Pipeline complete!"
echo "  Results saved to: reports/results_tables/"
echo "========================================================"
