#!/usr/bin/env bash
# download_voxceleb.sh — Download VoxCeleb1 for speaker verification training.
#
# Usage:
#   bash data/download_voxceleb.sh [--dest datasets/VoxCeleb1]
#
# Official source: https://mm.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html
# Requires: registration on the VoxCeleb website to receive download credentials.
#
# The full dataset is ~39 GB (audio only).  For student-grade training,
# this script optionally downloads only a random 300-speaker subset,
# which is approximately 10-12 GB.

set -euo pipefail

DEST="${1:-datasets/VoxCeleb1}"
NUM_SPEAKERS="${NUM_SPEAKERS:-300}"  # Set to "" to download all 1251 speakers
mkdir -p "$DEST/wav"

echo "=== VoxCeleb1 Download ==="
echo "Destination:   $DEST"
echo "Num speakers:  ${NUM_SPEAKERS:-all}"
echo ""

# --------------------------------------------------------------------------
# Credentials — set after registering at:
#   https://mm.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html
# --------------------------------------------------------------------------
VGG_USER="${VGG_USER:-}"
VGG_PASS="${VGG_PASS:-}"

if [ -z "$VGG_USER" ] || [ -z "$VGG_PASS" ]; then
    echo "[!] Please set VGG_USER and VGG_PASS environment variables."
    echo "    Register at: https://mm.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html"
    echo ""
    echo "    Then run:"
    echo "    VGG_USER=youruser VGG_PASS=yourpass bash data/download_voxceleb.sh"
    exit 1
fi

# --------------------------------------------------------------------------
# Download audio zip files
# --------------------------------------------------------------------------
BASE_URL="https://mm.robots.ox.ac.uk/~vgg/data/voxceleb/data"

for PART in vox1_dev_wav_partaa vox1_dev_wav_partab vox1_dev_wav_partac vox1_dev_wav_partad; do
    echo "Downloading $PART..."
    wget --user="$VGG_USER" --password="$VGG_PASS" \
         -c "$BASE_URL/$PART" -O "$DEST/$PART" || {
        echo "[ERROR] Failed to download $PART. Check credentials."
        exit 1
    }
done

# Combine parts and extract
cat "$DEST"/vox1_dev_wav_part* > "$DEST/vox1_dev_wav.zip"
unzip -q "$DEST/vox1_dev_wav.zip" -d "$DEST"

# Download verification pairs list
echo "Downloading official verification pairs list..."
wget -c "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt" \
     -O "$DEST/veri_test.txt"

# --------------------------------------------------------------------------
# Optional: subsample to NUM_SPEAKERS speakers for faster training
# --------------------------------------------------------------------------
if [ -n "$NUM_SPEAKERS" ]; then
    echo "Subsampling to $NUM_SPEAKERS speakers..."
    # TODO: Python script to select random subset and move others to a 'held_out' dir
    python3 - <<EOF
import os, random, shutil
random.seed(42)
wav_dir = "$DEST/wav"
all_speakers = sorted(os.listdir(wav_dir))
keep = set(random.sample(all_speakers, min($NUM_SPEAKERS, len(all_speakers))))
held_out_dir = "$DEST/held_out_speakers"
os.makedirs(held_out_dir, exist_ok=True)
for spk in all_speakers:
    if spk not in keep:
        shutil.move(os.path.join(wav_dir, spk), os.path.join(held_out_dir, spk))
print(f"Kept {len(keep)} speakers in {wav_dir}")
EOF
fi

echo ""
echo "VoxCeleb1 download complete.  wav/ directory: $DEST/wav"
echo "Verification pairs: $DEST/veri_test.txt"
