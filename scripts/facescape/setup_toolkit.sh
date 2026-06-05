#!/usr/bin/env bash
# Bootstrap the FaceScape toolkit into third_party/ (git-ignored, not vendored).
# Clones the upstream repo at a pinned commit and downloads its public sample data
# so the render pipeline (scripts/data/render_rgbd.py) and predef files
# (landmark_indices.npz, Rt_scale_dict.json) are available locally.
#
# Usage:  bash scripts/data/setup_toolkit.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$REPO_ROOT/third_party/facescape_toolkit"
PINNED_COMMIT="6a43878cdb61834472eb6cac7009d91f14a972d4"
SAMPLE_URL="https://box.nju.edu.cn/f/e33302f9b9ce4c7597d0/?dl=1"

mkdir -p "$REPO_ROOT/third_party"

if [ ! -d "$DEST/.git" ]; then
  echo "[setup] cloning facescape toolkit -> $DEST"
  git clone https://github.com/zhuhao-nju/facescape.git "$DEST"
else
  echo "[setup] toolkit already cloned, fetching"
  git -C "$DEST" fetch --quiet origin
fi
git -C "$DEST" checkout --quiet "$PINNED_COMMIT"
echo "[setup] toolkit pinned at $PINNED_COMMIT"

# Public sample (textured TU model + a multi-view tuple) for the render self-test.
SAMPLES_DIR="$DEST/samples"
if [ ! -f "$SAMPLES_DIR/sample_tu_model/1_neutral.obj" ]; then
  echo "[setup] downloading public sample data (~37 MB)"
  curl -fsSL "$SAMPLE_URL" -o "$SAMPLES_DIR/samples.tar.gz"
  tar -xzf "$SAMPLES_DIR/samples.tar.gz" -C "$SAMPLES_DIR" -k
  echo "[setup] sample data extracted"
else
  echo "[setup] sample data already present"
fi

echo "[setup] done."
