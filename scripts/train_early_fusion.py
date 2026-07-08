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
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from _init_paths import REPO_ROOT
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
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="cap #subjects (0=all) for quick runs")
    p.add_argument("--no-depth", action="store_true", help="RGB-only ablation arm")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    errs, n = 0.0, 0
    for batch in loader:
        batch = move(batch, device)
        hw = (batch["rgbd"].shape[-2], batch["rgbd"].shape[-1])
        preds_3d, _ = model(batch["rgbd"], batch["proj"], hw)
        b = batch["rgbd"].shape[0]
        errs += mpjpe_mm(preds_3d[-1], batch["landmarks_3d"]) * b
        n += b
    return errs / max(n, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    subs = discover_subjects(args.root)
    if args.limit:
        subs = subs[: args.limit]
    train_ids, val_ids = subject_disjoint_split(subs, args.val_frac, args.seed)
    print(f"subjects: {len(subs)}  train {len(train_ids)}  val {len(val_ids)}  "
          f"depth={'OFF' if args.no_depth else 'ON'}")

    train_ds = MultiViewFaceScape(args.root, train_ids)
    val_ds = MultiViewFaceScape(args.root, val_ids)
    train_ld = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                          num_workers=args.workers, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                        num_workers=args.workers)

    model = MultiViewLandmark3D(args.assets, num_layers=args.num_layers,
                                pretrained=False, use_depth=not args.no_depth,
                                img_size=args.img_size).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for it, batch in enumerate(train_ld):
            batch = move(batch, args.device)
            hw = (batch["rgbd"].shape[-2], batch["rgbd"].shape[-1])
            preds_3d, preds_2d = model(batch["rgbd"], batch["proj"], hw)
            losses = decoder_losses(preds_3d, preds_2d, batch["landmarks_3d"],
                                    batch["landmarks_2d"], batch["vis"])
            opt.zero_grad()
            losses["total"].backward()
            opt.step()
            running += losses["total"].item()
        sched.step()

        val_mpjpe = evaluate(model, val_ld, args.device)
        print(f"epoch {epoch:3d}  train_loss {running/max(len(train_ld),1):8.3f}  "
              f"val_MPJPE {val_mpjpe:7.2f} mm  ({time.time()-t0:.0f}s)")

        ckpt = {"model": model.state_dict(), "epoch": epoch, "val_mpjpe": val_mpjpe,
                "args": vars(args)}
        torch.save(ckpt, out / "last.pth")
        if val_mpjpe < best:
            best = val_mpjpe
            torch.save(ckpt, out / "best.pth")
    print(f"done. best val MPJPE {best:.2f} mm  ->  {out/'best.pth'}")


if __name__ == "__main__":
    main()
