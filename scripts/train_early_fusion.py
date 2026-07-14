"""Phase 7 - train the early-fusion multi-view 3D-landmark model on FaceScape.

Subject-disjoint split, from-scratch backbone, deep-supervised losses. Reports
held-out MPJPE (mean per-joint 3D error, mm). The RGB-vs-RGBD ablation is the
--no-depth flag (same everything, depth channel zeroed).

Example:
    .venv/bin/python scripts/train_early_fusion.py \
        --root data/facescape/virtual_camera_data --epochs 40 --bs 2 --lr 1e-4

Run it yourself (GPU strongly recommended). On Great Lakes, wrap in the usual
sbatch; locally, use a small --bs and --img-size 256.
"""

import _init_paths  # noqa: F401
import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from _init_paths import REPO_ROOT
from multi_view.data.augment import AugConfig, MultiViewAugmentor
from multi_view.data.facescape_dataset import (
    MultiViewFaceScape, discover_subjects, subject_disjoint_split)
from multi_view.losses import decoder_losses, mpjpe_mm
from multi_view.mv_model import MultiViewLandmark3D


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(REPO_ROOT / "data/facescape/virtual_camera_data"))
    p.add_argument("--assets", default=str(REPO_ROOT / "multi_view/assets"))
    p.add_argument("--out", default=str(REPO_ROOT / "output/early_fusion"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="max grad-norm for clipping; guards against NaN blow-ups")
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="cap #subjects (0=all) for quick runs")
    p.add_argument("--no-depth", action="store_true", help="RGB-only ablation arm")
    # iter-2 "messy data" domain randomization (applied to BOTH splits: train random,
    # val fixed-per-sample). Only RGB is augmented; depth + GT stay clean.
    p.add_argument("--bg-dir", default=str(REPO_ROOT / "data/backgrounds/indoor/Images"),
                   help="background image pool for compositing")
    p.add_argument("--bg-prob", type=float, default=0.0,
                   help="per-view prob of compositing a random background (0=off)")
    p.add_argument("--photometric", action="store_true",
                   help="apply HRNet's photometric ISP pipeline per view "
                        "(brightness/contrast/saturation/hue/blur/downscale/noise/JPEG)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (val_loss, val_mpjpe_mm) averaged over the loader (per-sample)."""
    model.eval()
    loss_sum, err_sum, n = 0.0, 0.0, 0
    for batch in loader:
        batch = move(batch, device)
        hw = (batch["rgbd"].shape[-2], batch["rgbd"].shape[-1])
        preds_3d, preds_2d = model(batch["rgbd"], batch["proj"], hw)
        b = batch["rgbd"].shape[0]
        losses = decoder_losses(preds_3d, preds_2d, batch["landmarks_3d"],
                                batch["landmarks_2d"], batch["vis"])
        loss_sum += float(losses["total"]) * b
        err_sum += float(mpjpe_mm(preds_3d[-1], batch["landmarks_3d"])) * b
        n += b
    n = max(n, 1)
    return loss_sum / n, err_sum / n


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # Persist per-epoch metrics: metrics.csv (structured) + train.log (raw console).
    logf = open(out / "train.log", "w")

    def logprint(msg):
        print(msg)
        logf.write(msg + "\n"); logf.flush()

    csvf = open(out / "metrics.csv", "w", newline="")
    writer = csv.writer(csvf)
    writer.writerow(["epoch", "lr", "train_loss", "val_loss", "val_mpjpe",
                     "skipped", "sec"])
    csvf.flush()

    subs = discover_subjects(args.root)
    if args.limit:
        subs = subs[: args.limit]
    train_ids, val_ids = subject_disjoint_split(subs, args.val_frac, args.seed)
    aug_cfg = AugConfig(bg_dir=args.bg_dir, bg_prob=args.bg_prob,
                        photometric=args.photometric)
    aug_note = (f"  aug[bg={args.bg_prob} photometric={args.photometric}]"
                if aug_cfg.enabled else "")
    logprint(f"subjects: {len(subs)}  train {len(train_ids)}  val {len(val_ids)}  "
             f"depth={'OFF' if args.no_depth else 'ON'}{aug_note}")

    # One augmentor instance (shared bg pool); train = fresh randomness, val =
    # deterministic per-sample so the held-out metric is stable across epochs.
    augmentor = MultiViewAugmentor(aug_cfg) if aug_cfg.enabled else None
    train_ds = MultiViewFaceScape(args.root, train_ids, augmentor=augmentor,
                                  aug_deterministic=False)
    val_ds = MultiViewFaceScape(args.root, val_ids, augmentor=augmentor,
                                aug_deterministic=True, aug_seed=args.seed)
    train_ld = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                          num_workers=args.workers, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                        num_workers=args.workers)

    model = MultiViewLandmark3D(args.assets, num_layers=args.num_layers,
                                use_depth=not args.no_depth,
                                img_size=args.img_size).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, skipped = time.time(), 0.0, 0
        for it, batch in enumerate(train_ld):
            batch = move(batch, args.device)
            hw = (batch["rgbd"].shape[-2], batch["rgbd"].shape[-1])
            preds_3d, preds_2d = model(batch["rgbd"], batch["proj"], hw)
            losses = decoder_losses(preds_3d, preds_2d, batch["landmarks_3d"],
                                    batch["landmarks_2d"], batch["vis"])
            loss = losses["total"]
            # Skip a batch whose loss is already non-finite (do not backward NaN).
            if not torch.isfinite(loss):
                skipped += 1
                continue
            opt.zero_grad()
            loss.backward()
            # Clip returns the pre-clip total grad norm; if the grads themselves are
            # non-finite (e.g. an SVD-backward NaN), skip the step so the optimizer
            # never writes NaN into the weights. This is what makes the run recover
            # from a single bad batch instead of collapsing permanently.
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if torch.isfinite(gnorm):
                opt.step()
                running += loss.item()
            else:
                skipped += 1
        lr = opt.param_groups[0]["lr"]
        sched.step()

        train_loss = running / max(len(train_ld) - skipped, 1)
        val_loss, val_mpjpe = evaluate(model, val_ld, args.device)
        sec = time.time() - t0
        skip_note = f"  skipped {skipped}" if skipped else ""
        logprint(f"epoch {epoch:3d}  train_loss {train_loss:8.3f}  "
                 f"val_loss {val_loss:8.3f}  val_MPJPE {val_mpjpe:7.2f} mm  "
                 f"({sec:.0f}s){skip_note}")
        writer.writerow([epoch, f"{lr:.3e}", f"{train_loss:.6f}",
                         f"{val_loss:.6f}", f"{val_mpjpe:.6f}", skipped,
                         f"{sec:.1f}"])
        csvf.flush()

        ckpt = {"model": model.state_dict(), "epoch": epoch, "val_mpjpe": val_mpjpe,
                "val_loss": val_loss, "args": vars(args)}
        torch.save(ckpt, out / "last.pth")
        if val_mpjpe < best:
            best = val_mpjpe
            torch.save(ckpt, out / "best.pth")
    logprint(f"done. best val MPJPE {best:.2f} mm  ->  {out/'best.pth'}")
    csvf.close(); logf.close()


if __name__ == "__main__":
    main()
