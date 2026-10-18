#!/usr/bin/env bash
# download_in_the_wild.sh — Download the In-the-Wild deepfake speech dataset.
#
# Usage:
#   bash data/download_in_the_wild.sh [--dest datasets/in-the-wild]
#
# Reference: Müller et al. (2022) "Does Audio Deepfake Detection Generalize?"
#
# IMPORTANT: The dataset host has changed at least once historically.
# This script verifies the URL is live before downloading.
# DO NOT hardcode a URL that was found in an old paper or GitHub repo —
# verify it resolves first.

set -euo pipefail

DEST="${1:-datasets/in-the-wild}"
mkdir -p "$DEST"

echo "=== In-the-Wild Dataset Download ==="
echo "Destination: $DEST"
echo ""

# --------------------------------------------------------------------------
# Verify URL is live before downloading
# --------------------------------------------------------------------------
CANDIDATE_URL="https://owncloud.fraunhofer.de/index.php/s/FZaR8RxMCg9BPAB"

echo "Checking if the dataset host is live..."
if curl --silent --head --fail "$CANDIDATE_URL" > /dev/null 2>&1; then
    echo "[OK] Host is live: $CANDIDATE_URL"
    DATASET_URL="$CANDIDATE_URL"
else
    echo "[ERROR] The candidate URL is not reachable: $CANDIDATE_URL"
    echo ""
    echo "The In-the-Wild dataset host has moved in the past."
    echo "Search for the current host using:"
    echo "  'Müller Does Audio Deepfake Detection Generalize dataset download'"
    echo "or check the paper's GitHub: https://github.com/piotrkawa/deepfake-whisper-features"
    echo ""
    echo "Once you find the correct URL, set DATASET_URL in this script and re-run."
    exit 1
fi

# --------------------------------------------------------------------------
# Download and extract
# --------------------------------------------------------------------------
echo "Downloading dataset..."
wget -c "$DATASET_URL/download" -O "$DEST/in-the-wild.tar.gz" || {
    echo "[ERROR] Download failed. The URL may have changed."
    echo "Please verify the current download URL and update this script."
    exit 1
}

echo "Extracting..."
tar -xzf "$DEST/in-the-wild.tar.gz" -C "$DEST" --strip-components=1

# --------------------------------------------------------------------------
# Create a meta.csv with file + label columns
# (The original dataset may have a different structure — adapt as needed)
# --------------------------------------------------------------------------
python3 - <<'EOF'
import os, csv, pathlib

dest = os.environ.get("DEST", "datasets/in-the-wild")
rows = []
for root, _, files in os.walk(dest):
    for f in files:
        if f.endswith(('.wav', '.mp3', '.flac')):
            rel_path = os.path.relpath(os.path.join(root, f), dest)
            # Convention: files under 'real/' or 'bonafide/' are bonafide;
            # files under 'fake/' or 'spoof/' are spoof.
            # Adapt this to the actual directory structure after extraction.
            if any(x in rel_path.lower() for x in ['real', 'bonafide', 'genuine']):
                label = 'bonafide'
            else:
                label = 'spoof'
            rows.append({'file': rel_path, 'label': label})

meta_path = os.path.join(dest, 'meta.csv')
with open(meta_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['file', 'label'])
    writer.writeheader()
    writer.writerows(rows)

print(f"Created {meta_path} with {len(rows)} files "
      f"({sum(1 for r in rows if r['label'] == 'bonafide')} bonafide, "
      f"{sum(1 for r in rows if r['label'] == 'spoof')} spoof)")
EOF

echo ""
echo "In-the-Wild dataset ready at: $DEST"
echo "NOTE: Verify the label assignment in meta.csv matches the actual"
echo "      directory structure extracted from the archive."
