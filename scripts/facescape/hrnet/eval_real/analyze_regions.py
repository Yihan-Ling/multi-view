#!/usr/bin/env python3
"""Decompose the inter-ocular NME by facial region, to localize WHERE a
synthetic-trained model fails on real images.

Overall NME (as HRNet computes it) == the mean over all 68 points of
    ||pred_i - gt_i|| / ||gt[36] - gt[45]||
so averaging that same per-point normalized error within iBUG-68 groups gives a
region breakdown that reconciles back to the headline number.

Reads through the same Face300W loader + HRNet decode_preds (no new eval logic).

Run from repo root (you run models, not me):
  PYTHONPATH=third_party/HRNet-Facial-Landmark-Detection \
  .venv/bin/python scripts/facescape/hrnet/eval_real/analyze_regions.py \
    --cfg scripts/facescape/hrnet/eval_real/face_alignment_aflw2000_w18.yaml \
    --model-file output/hrnet/forte_trained/300W/face_alignment_facescape_w18/model_best.pth
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import lib.models as models
from lib.config import config, update_config
from lib.datasets import get_dataset
from lib.core.evaluation import decode_preds

from run_eval import load_state_dict_any

# iBUG-68 regions (0-indexed, inclusive ranges)
REGIONS = {
    "jaw/contour": range(0, 17),
    "r_eyebrow":   range(17, 22),
    "l_eyebrow":   range(22, 27),
    "nose":        range(27, 36),
    "r_eye":       range(36, 42),
    "l_eye":       range(42, 48),
    "mouth":       range(48, 68),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--model-file", required=True)
    args = ap.parse_args()
    update_config(config, args)

    config.defrost(); config.MODEL.INIT_WEIGHTS = False; config.freeze()
    model = models.get_face_alignment_net(config)
    model = nn.DataParallel(model, device_ids=list(config.GPUS)).cuda()
    model.module.load_state_dict(load_state_dict_any(args.model_file))
    model.eval()

    ds = get_dataset(config)(config, is_train=False)
    loader = DataLoader(ds, batch_size=config.TEST.BATCH_SIZE_PER_GPU,
                        shuffle=False, num_workers=config.WORKERS)
    res = config.MODEL.HEATMAP_SIZE

    # accumulate per-point normalized error over the whole set
    per_point = np.zeros(config.MODEL.NUM_JOINTS)
    n = 0
    with torch.no_grad():
        for inp, target, meta in loader:
            out = model(inp.cuda())
            preds = decode_preds(out.cpu(), meta["center"], meta["scale"], res).numpy()
            gt = meta["pts"].numpy()
            interocular = np.linalg.norm(gt[:, 36] - gt[:, 45], axis=1)  # (B,)
            d = np.linalg.norm(preds - gt, axis=2)                        # (B,68)
            per_point += (d / interocular[:, None]).sum(axis=0)
            n += preds.shape[0]
    per_point /= n  # mean normalized error per landmark

    print(f"\nsamples: {n}   overall NME: {per_point.mean():.4f}\n")
    print(f"{'region':12s} {'NME':>8s}  {'#pts':>4s}")
    for name, idx in REGIONS.items():
        idx = list(idx)
        print(f"{name:12s} {per_point[idx].mean():8.4f}  {len(idx):>4d}")

    # Aggregates that drop the hardest regions, so the headline number isn't
    # dominated by the contour (which the synthetic cap/no-hair renders never
    # taught and which the benchmarks define ambiguously). These average the same
    # per-point normalized error over a SUBSET of points.
    inner = list(range(17, 68))            # all but jaw/contour (51 pts)
    core = list(range(27, 68))             # nose+eyes+mouth only (41 pts)
    print()
    print(f"{'inner-51':12s} {per_point[inner].mean():8.4f}  {len(inner):>4d}  (no jaw/contour)")
    print(f"{'core-41':12s} {per_point[core].mean():8.4f}  {len(core):>4d}  (no contour/brows)")


if __name__ == "__main__":
    main()
