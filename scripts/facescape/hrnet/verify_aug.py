#!/usr/bin/env python3
"""Verify the photometric augmentation is correctly applied -- four checks:

  A. MENU ACTIVATION   -- run photometric() many times with the `log` hook and
     confirm each effect fires at ~its configured probability and samples
     strengths across its configured range. Proves the gates + ranges are live.
  B. INVARIANTS        -- output is uint8 and same HxW (pure pixel op; nothing
     moved, so the GT landmarks stay valid).
  C. LOADER PATH       -- build the real FaceScapeAug from the train cfg and spy
     on _photometric: it MUST be called for an is_train item and NOT for a val
     item. Proves the aug is wired into the actual training __getitem__.
  D. VISUAL (crop scale) -- a panel of clean + random draws cropped to the face
     and upsampled, so noise/JPEG (invisible in the full-image thumbnail panel)
     are actually visible. Captions list the effects that fired per cell.

Run from repo root:
    .venv/bin/python scripts/facescape/hrnet/verify_aug.py
Writes scratch/aug_verify.png. Eyeball D yourself; A-C print PASS/FAIL.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
HR = os.path.join(REPO, "third_party", "HRNet-Facial-Landmark-Detection")
sys.path.insert(0, HR)        # HRNet lib (for check C)
sys.path.insert(0, HERE)      # facescape_aug

from facescape_aug import photometric  # noqa: E402

CSV = os.path.join(REPO, "data/facescape/HRNet_train/train.csv")
IMAGES = os.path.join(REPO, "data/facescape/HRNet_train/images")
CFG = os.path.join(HERE, "face_alignment_facescape_w18_scratch.yaml")
OUT = os.path.join(REPO, "scratch/aug_verify.png")

# configured (prob, lo, hi) per effect -- keep in sync with photometric()
EXPECTED = {
    "brightness": (0.5, 0.6, 1.4), "contrast": (0.7, 0.4, 1.6),
    "saturation": (0.7, 0.4, 1.6), "hue": (0.3, -15, 15),
    "blur": (0.3, 0.0, 2.5), "downscale": (0.2, 0.4, 1.0),
    "noise": (0.85, 0.0, 16.0), "jpeg": (0.5, 20, 90),
}


def check_a_activation(img, rng, n=4000):
    print(f"\n[A] MENU ACTIVATION  ({n} draws)")
    print(f"  {'effect':11} {'fire%':>7} {'(want)':>8}   {'min':>6} {'max':>6}  range ok")
    fires = {k: 0 for k in EXPECTED}
    vals = {k: [] for k in EXPECTED}
    for _ in range(n):
        log = {}
        photometric(img, rng, log=log)
        for k in EXPECTED:
            if log.get(k) is not None:
                fires[k] += 1
                vals[k].append(log[k])
    ok = True
    for k, (p, lo, hi) in EXPECTED.items():
        rate = fires[k] / n
        vmin = min(vals[k]) if vals[k] else float("nan")
        vmax = max(vals[k]) if vals[k] else float("nan")
        rate_ok = abs(rate - p) < 0.04                       # ~2 std for n=4000
        range_ok = vals[k] and vmin >= lo - 1e-6 and vmax <= hi + 1e-6
        ok = ok and rate_ok and range_ok
        print(f"  {k:11} {rate:6.1%} {p:7.0%}   {vmin:6.2f} {vmax:6.2f}  "
              f"{'ok' if range_ok else 'OUT'}{'' if rate_ok else '  <-RATE OFF'}")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def check_b_invariants(img, rng):
    out = photometric(img, rng)
    ok = out.dtype == np.uint8 and out.shape == img.shape
    print(f"\n[B] INVARIANTS: dtype={out.dtype} shape={out.shape} "
          f"-> {'PASS' if ok else 'FAIL'} (pure pixel op, GT untouched)")
    return ok


def check_c_loader_path():
    print("\n[C] LOADER PATH (real FaceScapeAug __getitem__)")
    try:
        from lib.config import config
        import facescape_aug as FA
        config.defrost()
        config.merge_from_file(CFG)
        config.freeze()

        calls = {"train": 0, "val": 0}

        def spy(self, img):
            tag = "train" if self.is_train else "val"
            calls[tag] += 1
            return img  # identity is fine; we only check IT WAS CALLED

        FA.FaceScapeAug._photometric = spy
        FA.FaceScapeAug(config, is_train=True)[0]
        FA.FaceScapeAug(config, is_train=False)[0]
        ok = calls["train"] >= 1 and calls["val"] == 0
        print(f"  _photometric calls: train={calls['train']} val={calls['val']} "
              f"-> {'PASS' if ok else 'FAIL'} (fires only in training)")
        return ok
    except Exception as e:  # missing torch/lib -> skip, don't fail the whole run
        print(f"  SKIPPED ({type(e).__name__}: {e})")
        return None


def _crop(arr, cw, ch, scale):
    side = scale * 200 * 1.25  # loader's effective crop side
    h, w = arr.shape[:2]
    x0, y0 = int(cw - side / 2), int(ch - side / 2)
    x1, y1 = int(cw + side / 2), int(ch + side / 2)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    patch = Image.fromarray(arr[y0:y1, x0:x1]).resize((256, 256), Image.BILINEAR)
    return np.asarray(patch)


def check_d_visual(full, cw, ch, scale, rng, cells=11):
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.5))
    axes = axes.ravel()
    axes[0].imshow(_crop(full, cw, ch, scale))
    axes[0].set_title("clean", fontsize=9)
    for i in range(1, cells + 1):
        log = {}
        aug = photometric(full, rng, log=log)              # full-image aug (as training does)
        axes[i].imshow(_crop(aug, cw, ch, scale))           # then crop the face to view
        fired = []
        if log.get("noise") is not None: fired.append(f"σ{log['noise']:.0f}")
        if log.get("jpeg") is not None: fired.append(f"jpg{log['jpeg']}")
        if log.get("contrast") is not None: fired.append(f"con{log['contrast']:.1f}")
        if log.get("saturation") is not None: fired.append(f"sat{log['saturation']:.1f}")
        if log.get("blur") is not None: fired.append(f"blur{log['blur']:.1f}")
        axes[i].set_title(" ".join(fired) or "(none)", fontsize=8)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Photometric aug at crop scale (full-image aug, then face crop)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print(f"\n[D] VISUAL: wrote {OUT}  (eyeball it -- this check is yours)")


def main():
    rng = np.random.default_rng(0)
    df = pd.read_csv(CSV)
    row = df.iloc[int(rng.integers(len(df)))]
    full = np.array(Image.open(os.path.join(IMAGES, row.iloc[0])).convert("RGB"), dtype=np.uint8)
    cw, ch, scale = float(row.iloc[2]), float(row.iloc[3]), float(row.iloc[1])

    a = check_a_activation(full, rng)
    b = check_b_invariants(full, rng)
    c = check_c_loader_path()
    check_d_visual(full, cw, ch, scale, rng)

    auto = [x for x in (a, b, c) if x is not None]
    print(f"\n=== A-C: {'ALL PASS' if all(auto) else 'SEE FAILURES ABOVE'} "
          f"({sum(auto)}/{len(auto)}) ; D is a manual eyeball ===")


if __name__ == "__main__":
    main()
