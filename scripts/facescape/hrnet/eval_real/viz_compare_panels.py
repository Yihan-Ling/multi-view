#!/usr/bin/env python3
"""Build one pred-overlay PANEL per model (red = GT, green = prediction) on the
SAME real AFLW2000-3D faces, so the three sim-to-real iterations are visually
comparable. Each panel's suptitle carries the model's overall inter-ocular NME
(computed here via HRNet's own function.inference, not hardcoded).

Style matches scratch/pred_overlay_panel.png: a 2x3 grid of overlays.

Run from repo root (you run models, not me):
  PYTHONPATH=third_party/HRNet-Facial-Landmark-Detection \
  .venv/bin/python scripts/facescape/hrnet/eval_real/viz_compare_panels.py \
    [--cfg <eval yaml>] [--suffix _wflw]
Writes scratch/pred_panel_<tag><suffix>.png  (one per model)
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import lib.models as models
from lib.config import config, update_config
from lib.datasets import get_dataset
from lib.core import function

from run_eval import load_state_dict_any  # robust checkpoint loader

DEFAULT_CFG = "scripts/facescape/hrnet/eval_real/face_alignment_aflw2000_w18.yaml"
OUTDIR = "scratch"
N_SAMPLES = 6  # 2x3 panel

# (tag, human title, checkpoint) -- title is what the user asked for
MODELS = [
    ("photo_aug", "Photometric + bg (eye-leak bug)  [real NME 0.1462]",
     "output/hrnet/photo_aug/model_best.pth"),
    ("eyeblack_bg", "Photometric + bg + black eyes  [real NME 0.1964]",
     "output/hrnet/iter3_eyeblack_bg/model_best.pth"),
    ("nobg", "Photometric only, no bg  [real NME 0.2325]",
     "output/hrnet/iter2_nobg/model_best.pth"),
]


class _Args:
    cfg = DEFAULT_CFG
    model_file = ""  # set per model so update_config is happy


def per_image_nme(pred, gt):
    """68-pt inter-ocular NME for one face (matches lib compute_nme)."""
    interocular = np.linalg.norm(gt[36] - gt[45])
    return np.sum(np.linalg.norm(pred - gt, axis=1)) / (interocular * 68)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=DEFAULT_CFG, help="eval yaml (picks the dataset)")
    ap.add_argument("--suffix", default="", help="appended to output filenames, e.g. _wflw")
    cli = ap.parse_args()
    _Args.cfg = cli.cfg

    update_config(config, _Args)
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    config.defrost(); config.MODEL.INIT_WEIGHTS = False; config.freeze()

    # dataset / loader built ONCE -- identical across models
    ds = get_dataset(config)(config, is_train=False)
    gpus = list(config.GPUS)
    loader = DataLoader(
        dataset=ds, batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False, num_workers=config.WORKERS, pin_memory=config.PIN_MEMORY,
    )
    idxs = np.linspace(0, len(ds) - 1, N_SAMPLES).astype(int)

    # cache GT + image path for the sampled faces (same for every model)
    samples = []
    for idx in idxs:
        _, _, meta = ds[int(idx)]
        gt = meta["pts"].numpy()
        fname = ds.landmarks_frame.iloc[int(idx), 0]
        img = np.array(Image.open(os.path.join(config.DATASET.ROOT, fname)).convert("RGB"))
        samples.append((int(idx), gt, img, fname))

    for tag, title, ckpt in MODELS:
        model = models.get_face_alignment_net(config)
        model = nn.DataParallel(model, device_ids=gpus).cuda()
        model.module.load_state_dict(load_state_dict_any(ckpt))
        model.eval()

        nme, predictions = function.inference(config, loader, model)  # (N,68,2) image coords
        predictions = predictions.numpy()

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for ax, (idx, gt, img, fname) in zip(axes.ravel(), samples):
            pred = predictions[idx]
            pnme = per_image_nme(pred, gt)
            ax.imshow(img)
            ax.scatter(gt[:, 0], gt[:, 1], s=12, c="red", edgecolors="k", linewidths=0.2)
            ax.scatter(pred[:, 0], pred[:, 1], s=12, c="lime", edgecolors="k", linewidths=0.2)
            ax.set_title(f"NME={pnme:.3f}", fontsize=11)
            ax.axis("off")
        dsname = os.path.basename(config.DATASET.TESTSET)
        fig.suptitle(
            f"{title}  |  {dsname} overall NME = {nme:.4f}"
            f"   (red = ground truth, green = prediction)", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = os.path.join(OUTDIR, f"pred_panel_{tag}{cli.suffix}.png")
        fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"wrote {out}   overall NME={nme:.4f}")


if __name__ == "__main__":
    main()
