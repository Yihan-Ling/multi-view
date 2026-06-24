#!/usr/bin/env python3
"""Make one panel per training ROUND showing the data regime that round saw.

Three rounds, identical HRNetV2-W18 architecture -- the ONLY thing that changed
across them is how the FaceScape renders were augmented before the model saw them:

  sharp     -- clean renders, NO background, no photometric jitter
  forte     -- renders composited over random indoor backgrounds (depth>0 matte, ~80%)
  photo_aug -- forte's bg composite PLUS live photometric jitter (noise/color/JPEG/blur),
               trained from scratch

The original per-round image bundles were overwritten, so this regenerates each
regime from the SAME source renders (virtual_camera_data: rgb + depth + landmarks)
using the exact functions the rounds used -- composite_over_bg() and photometric().
Same faces as rows across all three panels => only the augmentation differs.

Run from repo root:
    .venv/bin/python scripts/facescape/hrnet/viz_round_data.py
Writes scratch/round_sharp.png, scratch/round_forte.png, scratch/round_photo_aug.png.
Eyeball them yourself -- this script does not judge.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "scripts", "facescape"))

from facescape_aug import photometric                       # noqa: E402
from build_hrnet_landmark_dataset import (                  # noqa: E402
    composite_over_bg, load_background_paths)

SRC = os.path.join(REPO, "data/facescape/virtual_camera_data")
BG_ROOT = os.path.join(REPO, "data/backgrounds/indoor/Images")
OUT_DIR = os.path.join(REPO, "scratch")

# distinct (subject, cam) source faces to use as rows. base lighting (rgb.png).
FACES = [("301", "0"), ("305", "2"), ("312", "0"), ("330", "1")]
NVARIANTS = 4   # augmented columns (the regime is random => show a spread)
SEED = 1


def load_face(subject: str, cam: str):
    d = os.path.join(SRC, subject, cam)
    rgb = np.asarray(Image.open(os.path.join(d, "rgb.png")).convert("RGB"), dtype=np.uint8)
    depth = np.load(os.path.join(d, "depth.npy"))
    return rgb, depth


def panel(title: str, render_cell, faces, ncols, out_path, col_titles=None):
    """render_cell(rgb, depth, rng, col) -> uint8 image for row=face, col=index."""
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(len(faces), ncols, figsize=(2.6 * ncols, 2.6 * len(faces)))
    axes = np.array(axes).reshape(len(faces), ncols)
    for r, (subject, cam) in enumerate(faces):
        rgb, depth = load_face(subject, cam)
        for c in range(ncols):
            ax = axes[r, c]
            ax.imshow(render_cell(rgb, depth, rng, c))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0 and col_titles:
                ax.set_title(col_titles[c], fontsize=10)
        axes[r, 0].set_ylabel(f"{subject}/{cam}", fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    bg_paths = load_background_paths(__import__("pathlib").Path(BG_ROOT))
    if not bg_paths:
        sys.exit(f"no backgrounds found under {BG_ROOT}")

    def pick_bg(rng):
        return np.asarray(Image.open(bg_paths[rng.integers(len(bg_paths))]).convert("RGB"))

    # --- sharp: clean render, every column identical (no randomness) ----------
    panel(
        "Round 1 -- sharp  (clean renders, NO background, no photometric aug)",
        lambda rgb, depth, rng, c: rgb,
        FACES, 1, os.path.join(OUT_DIR, "round_sharp.png"),
        col_titles=["clean render"],
    )

    # --- forte: composite over random indoor bg (depth>0 matte) ---------------
    panel(
        "Round 2 -- forte  (renders composited over random indoor backgrounds)",
        lambda rgb, depth, rng, c: rgb if c == 0
        else composite_over_bg(rgb, depth, pick_bg(rng), rng),
        FACES, 1 + NVARIANTS, os.path.join(OUT_DIR, "round_forte.png"),
        col_titles=["clean"] + [f"bg {i}" for i in range(1, NVARIANTS + 1)],
    )

    # --- photo_aug: bg composite THEN live photometric jitter -----------------
    panel(
        "Round 3 -- photo_aug  (bg composite + live photometric jitter, from scratch)",
        lambda rgb, depth, rng, c: rgb if c == 0
        else photometric(composite_over_bg(rgb, depth, pick_bg(rng), rng), rng),
        FACES, 1 + NVARIANTS, os.path.join(OUT_DIR, "round_photo_aug.png"),
        col_titles=["clean"] + [f"aug {i}" for i in range(1, NVARIANTS + 1)],
    )


if __name__ == "__main__":
    main()
