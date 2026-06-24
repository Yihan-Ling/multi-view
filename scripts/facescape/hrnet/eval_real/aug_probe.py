#!/usr/bin/env python3
"""Augmentation probe: find WHICH real-image corruption breaks the sim-trained model.

Take the FaceScape synthetic val set (where forte scores NME 0.039) and apply each
of several real-world corruptions at increasing severity, measuring NME at each step.
Wherever NME jumps from ~0.04 toward the real-image level (~0.49) is the domain
property the model is fragile to -- i.e. exactly the augmentation that's missing from
training. No training here; pure diagnostic.

Reuses HRNet's decode_preds + compute_nme (no new eval logic). The corruption is
applied to the SAME un-normalized 256x256 crop the loader feeds the model, then
re-normalized, so it is an apples-to-apples perturbation of the real input.

Run from repo root:
  PYTHONPATH=third_party/HRNet-Facial-Landmark-Detection \
  .venv/bin/python scripts/facescape/hrnet/eval_real/aug_probe.py \
    --cfg scripts/facescape/hrnet/face_alignment_facescape_w18.yaml \
    --model-file output/hrnet/forte_trained/300W/face_alignment_facescape_w18/model_best.pth \
    --n 500
Writes scratch/aug_probe.png (NME-vs-severity curves) + prints a table.
"""
import argparse
import io
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts/facescape/hrnet/eval_real")
import lib.models as models
from lib.config import config, update_config
from lib.datasets import get_dataset
from lib.core.evaluation import decode_preds, compute_nme
from run_eval import load_state_dict_any

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# --- corruptions: each maps a [0,255] HWC uint8 crop + severity-param -> corrupted crop ---
def c_blur(im, r):       return im.filter(ImageFilter.GaussianBlur(radius=r)) if r else im
def c_jpeg(im, q):
    if q >= 100: return im
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=int(q)); buf.seek(0); return Image.open(buf).convert("RGB")
def c_downscale(im, f):
    if f >= 1.0: return im
    w, h = im.size; small = im.resize((max(1,int(w*f)), max(1,int(h*f))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)
def c_noise(im, s):
    if s <= 0: return im
    a = np.asarray(im, np.float32) + np.random.normal(0, s, (im.size[1], im.size[0], 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
def c_desat(im, amt):    return ImageEnhance.Color(im).enhance(1.0 - amt) if amt else im
def c_contrast(im, c):   return ImageEnhance.Contrast(im).enhance(c) if c != 1.0 else im

# severity sweeps (index 0 = identity baseline). label shown in the legend.
PROBES = {
    "gaussian_blur (radius)":   (c_blur,      [0, 1, 2, 3, 4]),
    "downscale (resolution x)": (c_downscale, [1.0, 0.5, 0.33, 0.25, 0.15]),
    "jpeg (quality)":           (c_jpeg,      [100, 50, 25, 12, 6]),
    "gaussian_noise (std)":     (c_noise,     [0, 8, 16, 32, 48]),
    "desaturate (amount)":      (c_desat,     [0, 0.25, 0.5, 0.75, 1.0]),
    "contrast (factor)":        (c_contrast,  [1.0, 0.75, 0.5, 0.35, 0.25]),
}


def crop_to_tensor(crop_uint8):
    """[0,255] HWC uint8 -> normalized CHW float32 (the model's exact input format)."""
    x = (crop_uint8.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--model-file", required=True)
    ap.add_argument("--n", type=int, default=500, help="how many val samples to probe (speed)")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    update_config(config, args)
    config.defrost(); config.MODEL.INIT_WEIGHTS = False; config.freeze()

    model = models.get_face_alignment_net(config)
    model = nn.DataParallel(model, device_ids=list(config.GPUS)).cuda()
    model.module.load_state_dict(load_state_dict_any(args.model_file)); model.eval()

    ds = get_dataset(config)(config, is_train=False)
    idxs = np.linspace(0, len(ds) - 1, min(args.n, len(ds))).astype(int)
    res = config.MODEL.HEATMAP_SIZE

    # Pre-extract the un-normalized crops + GT meta ONCE (corruptions re-use them).
    crops, centers, scales, pts = [], [], [], []
    for i in idxs:
        img, _, meta = ds[int(i)]
        crop = np.clip((img.transpose(1, 2, 0) * STD + MEAN) * 255, 0, 255).astype(np.uint8)
        crops.append(crop); centers.append(meta['center']); scales.append(meta['scale']); pts.append(meta['pts'])
    centers = torch.stack(centers); scales = torch.stack([torch.as_tensor(s) for s in scales])
    meta_all = {'pts': torch.stack(pts)}

    def nme_for(corrupt_fn, param):
        all_preds = []
        for b in range(0, len(crops), args.batch):
            batch = crops[b:b + args.batch]
            tens = torch.stack([crop_to_tensor(np.asarray(corrupt_fn(Image.fromarray(c), param)))
                                for c in batch]).cuda()
            with torch.no_grad():
                out = model(tens)
            preds = decode_preds(out, centers[b:b + args.batch], scales[b:b + args.batch], res)
            all_preds.append(preds)
        preds = torch.cat(all_preds)
        return float(np.mean(compute_nme(preds, meta_all)))

    print(f"probing {len(crops)} samples | baseline (severity 0) should be ~0.039\n")
    results = {}
    for name, (fn, params) in PROBES.items():
        row = [nme_for(fn, p) for p in params]
        results[name] = (params, row)
        print(f"{name:26s} " + "  ".join(f"{p}->{v:.3f}" for p, v in zip(params, row)))

    # plot: NME vs severity index, with the real-image collapse band for reference
    plt.figure(figsize=(9, 5.5))
    for name, (params, row) in results.items():
        plt.plot(range(len(row)), row, marker="o", label=name)
    plt.axhline(0.039, ls="--", c="green", lw=1, label="sim baseline (0.039)")
    plt.axhspan(0.40, 0.50, color="red", alpha=0.08, label="real-image collapse (~0.4-0.5)")
    plt.xlabel("severity (0 = clean sim  ->  4 = strongest)"); plt.ylabel("NME (inter-ocular)")
    plt.title("Aug probe: which corruption breaks the sim-trained model?")
    plt.legend(fontsize=8, loc="upper left"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("scratch/aug_probe.png", dpi=110)
    print("\nwrote scratch/aug_probe.png")


if __name__ == "__main__":
    main()
