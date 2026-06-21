#!/usr/bin/env python3
"""Diagnostic: draw a model's PREDICTED 68 landmarks (green) vs GT (red) on a few
real images, to see HOW the model fails when NME is high.

Reads samples through the SAME Face300W loader the eval uses, so the crop/framing
is identical to scoring. Reuses HRNet's decode_preds (no new eval logic).

Interpreting the output:
  - preds clustered near image center / collapsed to a mean face, ignoring the
    real face  -> DOMAIN COLLAPSE (synthetic-only model can't read real images).
  - preds form a face shape but shifted/scaled off the real face -> FRAMING bug.
  - preds land on the face but in scrambled order -> landmark-ORDER bug.

Run from repo root (you run models, not me):
  PYTHONPATH=third_party/HRNet-Facial-Landmark-Detection \
  .venv/bin/python scripts/facescape/hrnet/eval_real/viz_predictions.py \
    --cfg scripts/facescape/hrnet/eval_real/face_alignment_aflw2000_w18.yaml \
    --model-file output/hrnet/forte_trained/300W/face_alignment_facescape_w18/model_best.pth \
    --n 6
Writes scratch/pred_overlay_*.png
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import lib.models as models
from lib.config import config, update_config
from lib.datasets import get_dataset
from lib.core.evaluation import decode_preds

from run_eval import load_state_dict_any  # reuse the robust checkpoint loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--model-file", required=True)
    ap.add_argument("--n", type=int, default=6, help="how many images to draw")
    ap.add_argument("--outdir", default="scratch")
    args = ap.parse_args()
    update_config(config, args)

    config.defrost(); config.MODEL.INIT_WEIGHTS = False; config.freeze()
    model = models.get_face_alignment_net(config)
    model = nn.DataParallel(model, device_ids=list(config.GPUS)).cuda()
    model.module.load_state_dict(load_state_dict_any(args.model_file))
    model.eval()

    ds = get_dataset(config)(config, is_train=False)
    res = config.MODEL.HEATMAP_SIZE  # [64, 64]
    idxs = np.linspace(0, len(ds) - 1, args.n).astype(int)

    with torch.no_grad():
        for k, idx in enumerate(idxs):
            inp, target, meta = ds[idx]
            out = model(torch.from_numpy(inp).unsqueeze(0).cuda()
                        if isinstance(inp, np.ndarray) else inp.unsqueeze(0).cuda())
            center = meta["center"].unsqueeze(0)
            scale = torch.tensor([meta["scale"]])
            preds = decode_preds(out.cpu(), center, scale, res)[0].numpy()  # (68,2) image coords
            gt = meta["pts"].numpy()                                        # (68,2) image coords

            fname = ds.landmarks_frame.iloc[idx, 0]
            img = np.array(Image.open(os.path.join(config.DATASET.ROOT, fname)).convert("RGB"))

            fig, ax = plt.subplots(figsize=(7, 7))
            ax.imshow(img)
            ax.scatter(gt[:, 0], gt[:, 1], s=14, c="red", label="GT", edgecolors="k", linewidths=0.3)
            ax.scatter(preds[:, 0], preds[:, 1], s=14, c="lime", label="pred", edgecolors="k", linewidths=0.3)
            ax.legend(loc="lower right")
            ax.set_title(fname); ax.axis("off")
            out_path = os.path.join(args.outdir, f"pred_overlay_{k}.png")
            fig.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close(fig)
            print(f"wrote {out_path}  ({fname})")


if __name__ == "__main__":
    main()
