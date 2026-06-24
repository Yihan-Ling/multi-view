#!/usr/bin/env python3
"""Eyeball the on-the-fly photometric augmentation (facescape_aug.photometric).

Builds a panel: each ROW is one synthetic face from the HRNet train CSV; the
first COLUMN is the clean original, the rest are independent random augmentations.
GT landmarks (green) are overlaid on every cell -- since photometric() never moves
a pixel, the points should sit identically across a row. That is the visual proof
the GT stays aligned; the row variety is the proof the model sees a real spread of
appearances each epoch.

Run:
    .venv/bin/python scripts/facescape/hrnet/viz_aug.py
Output: scratch/aug_panel.png  (eyeball it yourself -- this script does not judge).
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# import photometric() without pulling in torch/HRNet (guarded import in the module)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from facescape_aug import photometric

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(REPO, "data/facescape/HRNet_train/train.csv"))
    ap.add_argument("--images", default=os.path.join(REPO, "data/facescape/HRNet_train/images"))
    ap.add_argument("--out", default=os.path.join(REPO, "scratch/aug_panel.png"))
    ap.add_argument("--rows", type=int, default=4, help="distinct faces")
    ap.add_argument("--augs", type=int, default=4, help="augmented variants per face")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.csv)
    pick = rng.choice(len(df), size=min(args.rows, len(df)), replace=False)

    ncols = 1 + args.augs
    fig, axes = plt.subplots(len(pick), ncols, figsize=(2.6 * ncols, 2.6 * len(pick)))
    axes = np.atleast_2d(axes)

    for r, idx in enumerate(pick):
        row = df.iloc[idx]
        img = np.array(Image.open(os.path.join(args.images, row.iloc[0])).convert("RGB"), dtype=np.uint8)
        pts = row.iloc[4:].values.astype(float).reshape(-1, 2)
        vis = pts[:, 1] > 0  # loader skips y<=0 sentinels (off-image chin pts)

        for c in range(ncols):
            ax = axes[r, c]
            shown = img if c == 0 else photometric(img, rng)
            ax.imshow(shown)
            ax.scatter(pts[vis, 0], pts[vis, 1], s=4, c="lime", edgecolors="none")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title("original" if c == 0 else f"aug {c}", fontsize=10)
        axes[r, 0].set_ylabel(row.iloc[0], fontsize=7)

    fig.suptitle("FaceScape photometric augmentation (GT landmarks in green)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
