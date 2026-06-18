#!/usr/bin/env bash
# One-shot setup for HRNet face-landmark training on the desktop.
# Run from the MAIN repo root:  bash scripts/facescape/hrnet/setup_hrnet.sh
#
# third_party/ is gitignored, so the HRNet repo is NOT in the clone -- this
# clones it, pins the known-good commit, applies the modern-scipy/numpy crop fix,
# and drops our config into the repo's experiments/ folder.
set -euo pipefail

HRNET_DIR="third_party/HRNet-Facial-Landmark-Detection"
HRNET_URL="https://github.com/HRNet/HRNet-Facial-Landmark-Detection.git"
HRNET_COMMIT="f776dbe"   # the commit this setup was written against
CFG="scripts/facescape/hrnet/face_alignment_facescape_w18.yaml"

if [ ! -d "$HRNET_DIR/.git" ]; then
  echo ">> cloning HRNet into $HRNET_DIR"
  git clone "$HRNET_URL" "$HRNET_DIR"
fi

echo ">> pinning commit $HRNET_COMMIT"
git -C "$HRNET_DIR" checkout -q "$HRNET_COMMIT"

echo ">> patching transforms.py (scipy.misc / np.math removal)"
python scripts/facescape/hrnet/fix_transforms.py

echo ">> placing config in experiments/facescape/"
mkdir -p "$HRNET_DIR/experiments/facescape"
cp "$CFG" "$HRNET_DIR/experiments/facescape/"

echo ">> creating pretrained-weights folder"
mkdir -p "$HRNET_DIR/hrnetv2_pretrained"

cat <<'DONE'

Setup complete. Remaining manual steps (see README.md):
  1. Copy the dataset from USB into  data/facescape/HRNet_train/
  2. (Recommended) download the W18 ImageNet weights into
     third_party/HRNet-Facial-Landmark-Detection/hrnetv2_pretrained/
  3. Train (from repo root):
     python third_party/HRNet-Facial-Landmark-Detection/tools/train.py \
       --cfg scripts/facescape/hrnet/face_alignment_facescape_w18.yaml
DONE
