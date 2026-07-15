"""Evaluate a trained early-fusion multi-view 3D-landmark checkpoint on its
held-out val subjects (or any subset of them).

The split and the model config are read back from the training run, so the eval
scores the EXACT subjects the model never saw and rebuilds the model with the
same knobs (num_layers / use_depth / img_size):

  - <ckpt_dir>/split.json  -> root + val_ids (frozen at train time)
  - ckpt["args"]           -> num_layers, no_depth, img_size, seed, aug config

Reports MPJPE (mean per-joint 3D error, mm) on the last decoder layer, matching
the number train_early_fusion.py logs as val_MPJPE.

Examples:
    # all held-out val subjects, matching the training-time augmentation
    .venv/bin/python scripts/eval_early_fusion.py --ckpt output/early_fusion/best.pth

    # a quick 20-subject sanity subset
    .venv/bin/python scripts/eval_early_fusion.py --ckpt output/early_fusion/best.pth --limit 20

    # specific identities, and per-subject breakdown
    .venv/bin/python scripts/eval_early_fusion.py --ckpt output/early_fusion/best.pth \
        --subjects 212_0 300_1 --per-subject

    # score on CLEAN RGB (ignore the messy-run augmentation)
    .venv/bin/python scripts/eval_early_fusion.py --ckpt output/early_fusion/best.pth --clean
"""

import _init_paths  # noqa: F401
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from _init_paths import REPO_ROOT
from multi_view.data.augment import AugConfig, MultiViewAugmentor
from multi_view.data.facescape_dataset import MultiViewFaceScape
from multi_view.losses import decoder_losses
from multi_view.model import MultiViewLandmark3D


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to best.pth / last.pth")
    p.add_argument("--split", default=None,
                   help="split.json (default: <ckpt_dir>/split.json)")
    p.add_argument("--assets", default=None,
                   help="decoder assets dir (default: the training-time --assets)")
    p.add_argument("--limit", type=int, default=0,
                   help="cap #val subjects (0=all); ignored if --subjects given")
    p.add_argument("--subjects", nargs="*", default=None,
                   help="explicit subject ids to score (must be a subset of val_ids)")
    p.add_argument("--clean", action="store_true",
                   help="score on clean RGB (disable the training-time augmentation)")
    p.add_argument("--per-subject", action="store_true",
                   help="print each subject's MPJPE, worst first")
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_val_ids(split, args):
    """Resolve which held-out subjects to score, guarding against leakage."""
    val_ids = split["val_ids"]
    if args.subjects:
        val_set = set(val_ids)
        bad = [s for s in args.subjects if s not in val_set]
        if bad:
            raise SystemExit(f"not in the val split (would leak): {bad}")
        return list(args.subjects)
    return val_ids[: args.limit] if args.limit else val_ids


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    ckpt_dir = ckpt_path.parent

    # weights_only=False: the checkpoint also carries the args dict, not just tensors.
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    a = ckpt["args"]  # vars(train args) frozen at save time

    split_path = Path(args.split) if args.split else ckpt_dir / "split.json"
    split = json.loads(split_path.read_text())

    val_ids = build_val_ids(split, args)
    root = split["root"]

    # Rebuild the SAME augmentation the training-time val metric used (deterministic
    # per-sample). --clean scores on untouched RGB instead.
    aug_cfg = AugConfig(bg_dir=a["bg_dir"], bg_prob=a["bg_prob"],
                        photometric=a["photometric"])
    augmentor = None if args.clean or not aug_cfg.enabled else MultiViewAugmentor(aug_cfg)
    val_ds = MultiViewFaceScape(root, val_ids, augmentor=augmentor,
                                aug_deterministic=True, aug_seed=a["seed"])
    val_ld = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                        num_workers=args.workers)

    assets = args.assets or a["assets"]
    model = MultiViewLandmark3D(assets, num_layers=a["num_layers"],
                                use_depth=not a["no_depth"],
                                img_size=a["img_size"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    aug_note = "clean" if augmentor is None else f"aug[bg={a['bg_prob']} photo={a['photometric']}]"
    print(f"ckpt {ckpt_path}  (epoch {ckpt.get('epoch','?')}, "
          f"train-time val_MPJPE {ckpt.get('val_mpjpe', float('nan')):.2f} mm)")
    print(f"scoring {len(val_ids)} val subjects from {split_path.name}  "
          f"depth={'OFF' if a['no_depth'] else 'ON'}  {aug_note}")

    # shuffle=False + drop_last=False -> loader yields subjects in val_ids order,
    # so a running counter maps per-sample errors back to subject ids.
    per_subject, loss_sum, err_sum, n = [], 0.0, 0.0, 0
    with torch.no_grad():
        for batch in val_ld:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            hw = (batch["rgbd"].shape[-2], batch["rgbd"].shape[-1])
            preds_3d, preds_2d = model(batch["rgbd"], batch["proj"], hw)
            b = batch["rgbd"].shape[0]

            losses = decoder_losses(preds_3d, preds_2d, batch["landmarks_3d"],
                                    batch["landmarks_2d"], batch["vis"])
            # per-sample MPJPE: L2 per joint, mean over the 68 joints -> (b,)
            err = (preds_3d[-1] - batch["landmarks_3d"]).norm(dim=-1).mean(dim=-1)

            loss_sum += float(losses["total"]) * b
            for e in err.tolist():
                per_subject.append((val_ids[n], e))
                err_sum += e
                n += 1

    n = max(n, 1)
    print(f"\nMPJPE  {err_sum / n:7.3f} mm   (mean over {n} subjects)")
    print(f"loss   {loss_sum / n:7.3f}      (deep-supervised total, for parity)")

    if args.per_subject:
        print("\nper-subject MPJPE (worst first):")
        for sid, e in sorted(per_subject, key=lambda x: -x[1]):
            print(f"  {sid:>10s}  {e:7.3f} mm")


if __name__ == "__main__":
    main()
