#!/usr/bin/env python3
"""Eyeball probe for the RetinaFace crop framing -- run this BEFORE the full bundle
build/train to (a) confirm RetinaFace fires on both synthetic renders and real
images, and (b) tune DET_PAD so the crop the model will actually see looks right.

For each input image it shows two panels:
  LEFT  : the full image with the RetinaFace box (green) and the HRNet crop square
          (yellow, side = scale*250 = the region the loader extracts after its
          internal scale*=1.25) centered on the box center.
  RIGHT : that crop square extracted + resized to 256x256 -- i.e. exactly what the
          network is fed.

Nothing here writes into the dataset; it only saves a panel PNG for you to judge
([[eyeball-checks-are-the-users]]). Pass a few SYNTHETIC renders and a few REAL
faces together so you can compare framing across the domain gap in one image.

Run from repo root, e.g.:
  .venv/bin/python scripts/facescape/hrnet/eval_real/viz_detector_crop.py \
    --images data/facescape/virtual_camera_data/*/0/rgb.png \
             data/AFLW2000/image00002.jpg data/WFLW/WFLW_images/*/*.jpg \
    --pad 1.15 --det-gpu 0 --out scratch/detector_crop_probe.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

# Shared helper (scripts/facescape/, two dirs up).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from face_detector_crop import detect_main_box, box_to_center_scale  # noqa: E402

LOADER_MULT = 1.25   # HRNet Face300W loader does scale *= 1.25 before cropping


def crop_square(img: np.ndarray, cx: float, cy: float, side: float, size: int = 256):
    """Extract the [cx,cy]-centered square of `side` px (clipped) -> size x size."""
    H, W = img.shape[:2]
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = int(round(cx + side / 2)), int(round(cy + side / 2))
    canvas = np.zeros((max(y1, H) - min(y0, 0), max(x1, W) - min(x0, 0), 3), np.uint8)
    oy, ox = -min(y0, 0), -min(x0, 0)          # offset of the real image in the canvas
    canvas[oy:oy + H, ox:ox + W] = img
    crop = canvas[y0 + oy:y1 + oy, x0 + ox:x1 + ox]
    return np.asarray(Image.fromarray(crop).resize((size, size), Image.BILINEAR))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", required=True, type=Path,
                    help="image paths (mix synthetic renders + real faces)")
    ap.add_argument("--pad", type=float, default=None,
                    help="override DET_PAD (crop context). Default = module DET_PAD.")
    ap.add_argument("--det-gpu", type=int, default=-1, help="-1 = CPU")
    ap.add_argument("--det-network", choices=["mobilenet", "resnet50"], default="mobilenet")
    ap.add_argument("--out", type=Path, default=Path("scratch/detector_crop_probe.png"))
    args = ap.parse_args()

    pad_kw = {} if args.pad is None else {"pad": args.pad}
    imgs = [p for p in args.images if p.exists()]
    if not imgs:
        raise SystemExit("no input images found")

    fig, axes = plt.subplots(len(imgs), 2, figsize=(8, 4 * len(imgs)))
    axes = np.atleast_2d(axes)
    for i, path in enumerate(imgs):
        img = np.asarray(Image.open(path).convert("RGB"))
        box = detect_main_box(img, gpu_id=args.det_gpu, network=args.det_network)
        ax_full, ax_crop = axes[i, 0], axes[i, 1]
        ax_full.imshow(img)
        ax_full.set_title(path.name, fontsize=8)
        ax_full.axis("off")
        if box is None:
            ax_full.set_title(f"{path.name}\nNO DETECTION", fontsize=8, color="red")
            ax_crop.axis("off")
            continue
        scale, cx, cy = box_to_center_scale(box, **pad_kw)
        side = scale * 200 * LOADER_MULT
        x0, y0, x1, y1 = box
        ax_full.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                    fill=False, edgecolor="lime", lw=2))
        ax_full.add_patch(Rectangle((cx - side / 2, cy - side / 2), side, side,
                                    fill=False, edgecolor="yellow", lw=2, ls="--"))
        ax_crop.imshow(crop_square(img, cx, cy, side))
        ax_crop.set_title(f"crop 256  (scale={scale:.3f})", fontsize=8)
        ax_crop.axis("off")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}  (green=RetinaFace box, yellow-dashed=HRNet crop square)")


if __name__ == "__main__":
    main()
