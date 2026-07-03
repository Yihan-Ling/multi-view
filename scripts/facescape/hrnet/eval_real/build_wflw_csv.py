#!/usr/bin/env python3
"""Convert WFLW (98-pt) -> the Face300W CSV format, so a FaceScape-trained
HRNetV2-W18 (68-pt, iBUG order) can be evaluated on it with the stock eval tools
(no model/eval code needed). Second real dataset after AFLW2000-3D.

This mirrors build_aflw2000_csv.py exactly, with ONE new piece: WFLW is natively
98-point, so we remap 98 -> 68 iBUG before writing the CSV. Everything else
(framing parity, CSV layout, 68-pt inter-ocular scoring) is reused.

=============================================================================
WHY THIS FILE EXISTS
-----------------------------------------------------------------------------
tools/test.py + lib/datasets/face300w.py + lib/core/evaluation.py already do the
whole eval. They only need a CSV in the 300W layout pointed at by DATASET.TESTSET.
So this script's ONLY job is: WFLW raw labels -> that CSV. Nothing else.

OUTPUT CSV LAYOUT (identical to build_aflw2000_csv.py / build_hrnet_landmark_dataset.py):
    header:  image, scale, center_w, center_h, p0, p1, ..., p135      (140 cols)
    row:     <rel/path/name.jpg>, <scale>, <cx>, <cy>, x0,y0, x1,y1, ... x67,y67
  - col 0 is the image path RELATIVE to DATASET.ROOT (= ./data/wflw/WFLW_images).
    The WFLW annotation line already carries that relative path ("0--Parade/foo.jpg"),
    so NO image copying is needed.
  - 68 points written in iBUG-68 order (remapped from the native 98).

=============================================================================
THE TWO VERIFY GATES (same as AFLW -- these make or break the NME)
-----------------------------------------------------------------------------
VERIFY #1  FRAMING PARITY. The model only ever saw faces cropped with the framing
    from build_hrnet_landmark_dataset.py:view2row(). We reproduce it by calling the
    SAME bbox_to_center_scale() the AFLW builder uses (imported below), derived from
    the 68 remapped points with the jaw EXCLUDED. We deliberately IGNORE WFLW's own
    detection rectangle -- using it would feed the model a scale it never trained on.

VERIFY #2  iBUG-68 ORDER + EYE CORNERS. compute_nme() hard-codes inter-ocular =
    ||pts[36] - pts[45]|| (outer eye corners). After the 98->68 remap, iBUG 36 must be
    the RIGHT outer eye corner (WFLW 60) and iBUG 45 the LEFT outer eye corner
    (WFLW 72). EYEBALL one overlay to confirm the remap before trusting any number
    ([[eyeball-checks-are-the-users]]).

=============================================================================
DATA FACTS (WFLW, from list_98pt_rect_attr_test.txt -- whitespace-separated lines)
-----------------------------------------------------------------------------
Each line = 207 tokens:
    [0:196]   98 landmarks as x0 y0 x1 y1 ... x97 y97   (float pixels, 0-indexed)
    [196:200] detection rect: x_min y_min x_max y_max    (UNUSED -- see VERIFY #1)
    [200:206] 6 attribute flags: pose expr illum makeup occlusion blur (0/1 ints)
              -> attr[0] == 1 means LARGE pose (use to optionally filter)
    [206]     image path relative to WFLW_images/, e.g. "0--Parade/0_..._116.jpg"

Get the files from https://wywu.github.io/projects/LAB/WFLW.html (ungated):
    WFLW_images.tar.gz      -> data/wflw/WFLW_images/<subdir>/<name>.jpg
    WFLW_annotations.tar.gz -> .../list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt
=============================================================================
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Reuse the AFLW builder's framing + writer so parity has a SINGLE source of truth.
# (Both files live in this dir; running this script directly puts it on sys.path.)
from build_aflw2000_csv import bbox_to_center_scale, write_csv

# Shared RetinaFace crop helper lives at scripts/facescape/ (two dirs up).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from face_detector_crop import detect_main_box  # noqa: E402
from face_detector_crop import box_to_center_scale as det_box_to_center_scale  # noqa: E402


# --- Canonical WFLW-98 -> iBUG-68 index map ----------------------------------
# 68 entries, in iBUG-68 output order; each value is the WFLW-98 source index.
# Eyes pick 6 of WFLW's 8 per-eye points; outer corners (WFLW 60 / 72) -> iBUG 36 / 45.
WFLW_98_TO_IBUG_68 = [
    # jaw / contour (iBUG 0-16): every other WFLW contour point (33 -> 17)
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
    33, 34, 35, 36, 37,                 # right eyebrow (iBUG 17-21)
    42, 43, 44, 45, 46,                 # left eyebrow  (iBUG 22-26)
    51, 52, 53, 54,                     # nose bridge   (iBUG 27-30)
    55, 56, 57, 58, 59,                 # nose bottom   (iBUG 31-35)
    60, 61, 63, 64, 65, 67,             # right eye      (iBUG 36-41)
    68, 69, 71, 72, 73, 75,             # left eye       (iBUG 42-47)
    76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,   # outer mouth (iBUG 48-59)
    88, 89, 90, 91, 92, 93, 94, 95,     # inner mouth   (iBUG 60-67)
]
assert len(WFLW_98_TO_IBUG_68) == 68


def parse_line(line: str) -> tuple[np.ndarray, int, str]:
    """One annotation line -> (pts68 in iBUG order [68,2], large_pose flag, rel_path)."""
    toks = line.split()
    pts98 = np.asarray(toks[:196], dtype=float).reshape(98, 2)
    large_pose = int(toks[200])          # attr[0]: 1 == large pose
    rel_path = toks[206]
    pts68 = pts98[WFLW_98_TO_IBUG_68]     # fancy-index remap, preserves order
    return pts68, large_pose, rel_path


def build_rows(anno_file: Path, img_root: Path, drop_large_pose: bool,
               crop_mode: str = "landmark", det_gpu_id: int = -1,
               det_network: str = "mobilenet") -> list[list]:
    rows: list[list] = []
    kept = skipped = 0
    n_fallback = 0
    for line in anno_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        pts, large_pose, rel_path = parse_line(line)
        if drop_large_pose and large_pose:
            skipped += 1
            continue
        if not (img_root / rel_path).exists():
            skipped += 1
            continue
        # VERIFY #1: framing MUST match the model's training crop. landmark =
        # jaw-excluded bbox (ignores WFLW's own rect); retinaface = detector box
        # on the real image (parity with a retinaface-cropped model).
        if crop_mode == "retinaface":
            img = np.asarray(Image.open(img_root / rel_path).convert("RGB"))
            box = detect_main_box(img, gpu_id=det_gpu_id, network=det_network)
            if box is not None:
                scale, cw, ch = det_box_to_center_scale(box)
            else:
                scale, cw, ch = bbox_to_center_scale(pts, exclude_jaw=True)
                n_fallback += 1
        else:
            scale, cw, ch = bbox_to_center_scale(pts, exclude_jaw=True)
        rows.append([rel_path, scale, cw, ch, *pts.flatten().tolist()])
        kept += 1
    if crop_mode == "retinaface":
        print(f"retinaface: {n_fallback}/{kept} images had no detection -> "
              f"fell back to landmark bbox")
    print(f"kept {kept} / skipped {skipped} (drop_large_pose={drop_large_pose})")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anno", type=Path,
                    default=Path("data/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/"
                                 "list_98pt_rect_attr_test.txt"),
                    help="WFLW 98-pt test annotation .txt")
    ap.add_argument("--img_root", type=Path, default=Path("data/WFLW/WFLW_images"),
                    help="dir the image paths in the anno are relative to (= DATASET.ROOT)")
    ap.add_argument("--out", type=Path, default=Path("data/WFLW/test_wflw68.csv"),
                    help="output CSV (must match the TESTSET in the eval yaml)")
    ap.add_argument("--drop_large_pose", action="store_true",
                    help="drop WFLW large-pose images (attr[0]==1) to roughly match the "
                         "synthetic frontal-ish ring, like AFLW's yaw<=45 filter; off by "
                         "default so the number matches the standard full-set WFLW benchmark")
    ap.add_argument("--crop-mode", choices=["landmark", "retinaface"], default="landmark",
                    help="Framing source. MUST match the model's training crop: use "
                         "'retinaface' to evaluate a model trained with "
                         "build_hrnet_landmark_dataset.py --crop-mode retinaface.")
    ap.add_argument("--det-gpu", type=int, default=-1,
                    help="GPU id for RetinaFace (-1 = CPU); only with --crop-mode retinaface")
    ap.add_argument("--det-network", choices=["mobilenet", "resnet50"], default="mobilenet")
    args = ap.parse_args()

    rows = build_rows(args.anno, args.img_root, args.drop_large_pose,
                      crop_mode=args.crop_mode, det_gpu_id=args.det_gpu,
                      det_network=args.det_network)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
