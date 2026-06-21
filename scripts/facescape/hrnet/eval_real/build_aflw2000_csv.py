#!/usr/bin/env python3
"""Convert AFLW2000-3D -> the Face300W CSV format, so a FaceScape-trained HRNet
can be evaluated on it with the stock tools/test.py (no model/eval code needed).

TEACHING SCAFFOLD -- structure + verified facts are given; you write the bodies
in the `# TODO(you)` gaps. Run nothing until the gaps are filled.

=============================================================================
WHY THIS FILE EXISTS
-----------------------------------------------------------------------------
tools/test.py + lib/datasets/face300w.py + lib/core/evaluation.py already do the
whole eval. They only need a CSV in the 300W layout pointed at by DATASET.TESTSET.
So this script's ONLY job is: AFLW2000-3D raw labels -> that CSV. Nothing else.

OUTPUT CSV LAYOUT (must match build_hrnet_landmark_dataset.py exactly):
    header:  image, scale, center_w, center_h, p0, p1, ..., p135      (140 cols)
    row:     <bare_filename.jpg>, <scale>, <cx>, <cy>, x0,y0, x1,y1, ... x67,y67
  - col 0 is the image filename RELATIVE to DATASET.ROOT (= ./data/AFLW2000).
    AFLW2000 keeps imageXXXXX.jpg and imageXXXXX.mat side by side, so the bare
    filename works and NO image copying is needed.
  - 68 points in iBUG-68 order (AFLW2000-3D is natively iBUG-68 -- no remap).

=============================================================================
THE TWO VERIFY GATES (these are what make or break the NME -- see memory)
-----------------------------------------------------------------------------
VERIFY #1  FRAMING PARITY. The model only ever saw faces cropped with the
    framing from build_hrnet_landmark_dataset.py:view2row(). We MUST reproduce
    the identical center/scale derivation here, or the model sees a scale it
    never trained on and NME is garbage for reasons unrelated to sim-to-real.
    The exact recipe is in `bbox_to_center_scale()` below -- keep it identical.

VERIFY #2  iBUG-68 ORDER + PIXEL CONVENTION. compute_nme() hard-codes inter-ocular
    = ||pts[36] - pts[45]|| (outer eye corners). AFLW2000-3D's pt3d_68 is already
    iBUG-68, so 36/45 should be the outer eye corners -- but EYEBALL one overlay to
    be sure before trusting any number ([[eyeball-checks-are-the-users]]).
    Pixel convention: our synthetic CSV is 0-indexed and scored fine; AFLW .mat is
    MATLAB-origin. A whole-face 1px shift is negligible for NME, but note your
    choice. (See transform math in face300w.py / evaluation.py if curious.)

=============================================================================
DATA FACTS (AFLW2000-3D, per image imageXXXXX.mat -- MATLAB v5, use scipy.io.loadmat)
-----------------------------------------------------------------------------
  m = scipy.io.loadmat(path)
  m['pt3d_68']    -> shape (3, 68) float. Rows 0,1 = x,y in image pixels (iBUG-68).
                     -> the 2D landmarks are pt3d_68[:2].T  -> (68, 2)
  m['Pose_Para']  -> shape (1, 7): [pitch, yaw, roll, t3dx, t3dy, t3dz, f] in RADIANS
                     -> yaw_deg = degrees(Pose_Para[0, 1]); use it to FILTER.

DEPENDENCY: scipy is NOT in .venv yet. Install once:
    .venv/bin/pip install scipy
=============================================================================
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io  # noqa: F401  (needed once you write load_landmarks_and_yaw)


# --- Framing constants: COPIED from build_hrnet_landmark_dataset.py:view2row ---
# Do NOT retune these -- parity with training is the whole point (VERIFY #1).
PAD = 1.25      # landmark boxes are tight (iBUG-68 has no forehead/ear pts); pad to fit the head
NUDGE = 0.08    # gentle upward bias for the missing forehead

# iBUG-68 outer eye corners, for the VERIFY #2 sanity check only.
LEFT_EYE_OUTER = 36
RIGHT_EYE_OUTER = 45


def bbox_to_center_scale(pts: np.ndarray, exclude_jaw: bool = False) -> tuple[float, float, float]:
    # Framing-parity test (VERIFY #1): the synthetic training crops derived their
    # bbox from chin-EXCLUDED landmarks (clipped jaw pts -> -1 sentinel). With
    # exclude_jaw=True we drop the jaw/contour pts (iBUG 0-16) from the bbox to
    # mimic that framing, while ALL 68 pts are still written for scoring.
    box_pts = pts[17:] if exclude_jaw else pts
    x0, y0 = box_pts.min(axis=0)
    x1, y1 = box_pts.max(axis=0)
    w = x1 - x0
    h = y1 - y0
    center_w = (x0 + x1) / 2
    center_h = (y0 + y1) / 2 - h * NUDGE
    scale = (w + h) / 2 / 200 * PAD
    return scale, center_w, center_h


def load_landmarks_and_yaw(mat_path: Path) -> tuple[np.ndarray, float]:
    m = scipy.io.loadmat(mat_path)
    pts = m['pt3d_68'][:2].T.astype(float)
    yaw_deg = math.degrees(float(m['Pose_Para'][0, 1]))
    return pts, yaw_deg


def build_rows(aflw_dir: Path, yaw_max: float, exclude_jaw: bool = False) -> list[list]:
    rows: list[list] = []
    kept = skipped = 0
    for mat_path in sorted(aflw_dir.glob("*.mat")):
        pts, yaw_deg = load_landmarks_and_yaw(mat_path)
        if abs(yaw_deg) > yaw_max:
            skipped += 1
            continue
        jpg = mat_path.with_suffix(".jpg")
        if not jpg.exists():
            skipped += 1
            continue
        scale, cw, ch = bbox_to_center_scale(pts, exclude_jaw=exclude_jaw)
        rows.append([jpg.name, scale, cw, ch, *pts.flatten().tolist()])
        kept+=1
    print(f"kept {kept} / skipped {skipped} (yaw_max={yaw_max})")
    return rows


def write_csv(rows: list[list], out_path: Path) -> None:
    cols = ["image", "scale", "center_w", "center_h"] + [f"p{i}" for i in range(136)]
    pd.DataFrame(rows, columns=cols).to_csv(out_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aflw_dir", type=Path, default=Path("data/AFLW2000"),
                    help="dir holding imageXXXXX.jpg + imageXXXXX.mat")
    ap.add_argument("--out", type=Path, default=Path("data/AFLW2000/test_yaw45.csv"),
                    help="output CSV (must match the TESTSET in the eval yaml)")
    ap.add_argument("--yaw_max", type=float, default=45.0,
                    help="keep |yaw| <= this (deg) to match the synthetic ~+-50 ring")
    # VERIFY #1 RESULT (2026-06-21): inner-face framing (drop jaw 0-16 from the
    # bbox) is the CORRECT parity with the synthetic chin-excluded crops -- it cut
    # NME by 0.10 (sharp) / 0.025 (forte) vs all-68 framing, so it is now the
    # default. --include_jaw restores the old all-68 bbox for comparison only.
    ap.add_argument("--include_jaw", action="store_true",
                    help="use the OLD all-68 bbox framing (jaw included); off by default "
                         "because it under-frames real faces vs training")
    args = ap.parse_args()

    rows = build_rows(args.aflw_dir, args.yaw_max, exclude_jaw=not args.include_jaw)
    write_csv(rows, args.out)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
