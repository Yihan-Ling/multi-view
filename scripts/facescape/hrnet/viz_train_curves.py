#!/usr/bin/env python3
"""Plot train loss + synthetic-val NME vs epoch for the three HRNet rounds.

Parses the vendored trainer's log lines:
    Train Epoch N ... loss:L nme:M
    Test  Epoch N ... loss:L nme:M [008]:.. [010]:..
("Test" here = the synthetic FaceScape val split, i.e. the validation curve.)

Run from repo root:
    .venv/bin/python scripts/facescape/hrnet/viz_train_curves.py
Writes scratch/train_curves.png.
"""
from __future__ import annotations

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

LOGS = {
    "plain":          "output/hrnet/sharp_trained/face_alignment_facescape_w18_2026-06-18-15-04_train.log",
    "background_aug": "output/hrnet/forte_trained/300W/face_alignment_facescape_w18/face_alignment_facescape_w18_2026-06-19-15-07_train.log",
    "photo_aug":      "output/hrnet/photo_aug/face_alignment_facescape_w18_scratch_2026-06-23-16-56_train.log",
}
COLORS = {"plain": "tab:red", "background_aug": "tab:orange", "photo_aug": "tab:green"}
LR_STEPS = [30, 50]

TR = re.compile(r"Train Epoch (\d+).*loss:([\d.]+) nme:([\d.]+)")
TE = re.compile(r"Test Epoch (\d+).*loss:([\d.]+) nme:([\d.]+)")


def parse(path):
    tr, te = {}, {}
    for line in open(os.path.join(REPO, path)):
        m = TR.search(line)
        if m:
            tr[int(m[1])] = (float(m[2]), float(m[3]))
        m = TE.search(line)
        if m:
            te[int(m[1])] = (float(m[2]), float(m[3]))
    ep_tr = sorted(tr)
    ep_te = sorted(te)
    return (ep_tr, [tr[e][0] for e in ep_tr],          # train loss
            ep_te, [te[e][1] for e in ep_te])           # val nme


def main():
    fig, (axL, axN) = plt.subplots(1, 2, figsize=(13, 5))
    for tag, path in LOGS.items():
        ep_tr, loss, ep_te, vnme = parse(path)
        c = COLORS[tag]
        axL.plot(ep_tr, loss, color=c, label=tag)
        axN.plot(ep_te, vnme, color=c, label=tag)
        best = min(range(len(vnme)), key=lambda i: vnme[i])
        axN.scatter([ep_te[best]], [vnme[best]], color=c, s=40, zorder=5)
        axN.annotate(f"{vnme[best]:.4f}", (ep_te[best], vnme[best]),
                     textcoords="offset points", xytext=(4, 6), fontsize=8, color=c)

    for ax in (axL, axN):
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    axL.set_yscale("log")
    axL.set_ylabel("train loss (heatmap MSE, log scale)")
    axL.set_title("Training loss vs epoch")
    axN.set_ylabel("synthetic-val NME (inter-ocular)")
    axN.set_title("Validation NME vs epoch  (dot = best checkpoint)")
    axN.set_ylim(0, 0.25)   # clip: early epochs + transient spikes run off-scale
    axN.legend(loc="center right")

    fig.suptitle("HRNetV2-W18 on FaceScape synthetic -- three augmentation rounds", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(REPO, "scratch/train_curves.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
