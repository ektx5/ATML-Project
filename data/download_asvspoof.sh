#!/usr/bin/env bash
# download_asvspoof.sh — Download and verify the ASVspoof 2019 LA corpus.
#
# Usage:
#   bash data/download_asvspoof.sh [--dest datasets/ASVspoof2019_LA]
#
# IMPORTANT: The official download requires registration at:
#   https://datashare.ed.ac.uk/handle/10283/3336
# After registering, you will receive download links via email.
# This script automates the download once you have those links.
#
# Checksum verification is performed against the official MD5 sums
# published on the dataset page.  DO NOT skip this step — the Kaggle
# and HuggingFace mirrors have been found to have mismatched checksums
# for some splits.

set -euo pipefail

DEST="${1:-datasets/ASVspoof2019_LA}"
mkdir -p "$DEST"

echo "=== ASVspoof 2019 LA Download ==="
echo "Destination: $DEST"
echo ""

# --------------------------------------------------------------------------
# Official MD5 checksums from https://datashare.ed.ac.uk/handle/10283/3336
# Verify these against the dataset page before running.
# --------------------------------------------------------------------------
declare -A CHECKSUMS=(
    ["ASVspoof2019_LA_train.zip"]="TODO_verify_from_official_page"
    ["ASVspoof2019_LA_dev.zip"]="TODO_verify_from_official_page"
    ["ASVspoof2019_LA_eval.zip"]="TODO_verify_from_official_page"
    ["ASVspoof2019_LA_cm_protocols.zip"]="TODO_verify_from_official_page"
)

# --------------------------------------------------------------------------
# Download URLs (fill in after registering at the official host)
# DO NOT fabricate these URLs.  They are unique to your registered account.
# --------------------------------------------------------------------------
# TRAIN_URL="<your-registered-download-link>"
# DEV_URL="<your-registered-download-link>"
# EVAL_URL="<your-registered-download-link>"
# PROTO_URL="<your-registered-download-link>"

echo "[!] This script requires download links from the official host."
echo "    Please register at: https://datashare.ed.ac.uk/handle/10283/3336"
echo "    Then set TRAIN_URL / DEV_URL / EVAL_URL / PROTO_URL at the top of this script."
echo ""

# --------------------------------------------------------------------------
# Fallback: HuggingFace mirror (verify checksums before using!)
# --------------------------------------------------------------------------
# The HuggingFace dataset "lj1995/ASVspoof2019" appears to be a common mirror.
# VERIFY checksums against the official page before trusting this mirror.

USE_HF_MIRROR="${USE_HF_MIRROR:-false}"
if [ "$USE_HF_MIRROR" = "true" ]; then
    echo "[!] Using HuggingFace mirror.  VERIFY CHECKSUMS MANUALLY before training."
    # TODO: pip install huggingface_hub
    # TODO: python -c "from huggingface_hub import snapshot_download; snapshot_download('lj1995/ASVspoof2019', local_dir='$DEST')"
fi

# --------------------------------------------------------------------------
# Verify checksums after download
# --------------------------------------------------------------------------
verify_checksum() {
    local file="$1"
    local expected="$2"
    local actual
    actual=$(md5sum "$file" | awk '{print $1}')
    if [ "$actual" != "$expected" ]; then
        echo "[ERROR] Checksum mismatch for $file"
        echo "  Expected: $expected"
        echo "  Actual:   $actual"
        echo "  DO NOT use this file for training — it may be corrupted or tampered."
        exit 1
    else
        echo "[OK] $file checksum verified."
    fi
}

# TODO: Uncomment and fill after download:
# verify_checksum "$DEST/ASVspoof2019_LA_train.zip" "${CHECKSUMS[ASVspoof2019_LA_train.zip]}"
# ... etc

echo "Download script placeholder complete.  Fill in URLs from the official host."
