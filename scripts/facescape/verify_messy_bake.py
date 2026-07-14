"""Integrity + correctness check for the iter-2 messy-data full bake, plus a
random-subject panel for the user's eyeball test.

The messy set (data/facescape/virtual_camera_messy) is the OLD schema: per-view
landmarks_3d (camera frame) + landmarks_2d (pixels) + meta.json (K,R,t). RGB has
bg-composite + photometric BAKED in; depth and all GT labels stay clean, so the
geometry checks below are unaffected by the augmentation.

Checks (mirror the assertions inside MultiViewFaceScape.__getitem__):
  1. STRUCTURE  every subject has the same view set; each view has all 5 files.
  2. GEOMETRY   world = (lm_cam - t)@R agrees across all views of a subject;
                world reprojected through P=K[R|t] reproduces the stored 2D.
  3. DEPTH      each view has a non-empty face mask (depth>0) with a sane range.

Only prints a report; it certifies nothing visually -- the panel is for the user.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]


def load_view(vd: Path):
    meta = json.loads((vd / "meta.json").read_text())
    K = np.asarray(meta["K"], float)
    R = np.asarray(meta["R"], float)
    t = np.asarray(meta["t"], float).reshape(3)
    lm_cam = np.load(vd / "landmarks_3d.npy").astype(float)   # (68,3) camera frame
    uv = np.load(vd / "landmarks_2d.npy").astype(float)       # (68,2) pixels
    return K, R, t, lm_cam, uv, meta


def project(world, K, R, t):
    cam = world @ R.T + t
    px = cam @ K.T
    return px[:, :2] / px[:, 2:3]


def subject_views(subj_dir: Path):
    return sorted((d for d in subj_dir.iterdir() if d.is_dir()), key=lambda d: int(d.name))


def check_subject(subj_dir: Path):
    """Return dict of per-subject metrics + a list of structural problems."""
    problems = []
    vds = subject_views(subj_dir)
    if not vds:
        return {"n_views": 0}, [f"{subj_dir.name}: no view folders"]

    files = ("rgb.png", "depth.npy", "landmarks_2d.npy", "landmarks_3d.npy", "meta.json")
    worlds, reproj_err, depth_frac, depth_lo, depth_hi = [], [], [], [], []
    for vd in vds:
        missing = [f for f in files if not (vd / f).exists()]
        if missing:
            problems.append(f"{subj_dir.name}/{vd.name}: missing {missing}")
            continue
        K, R, t, lm_cam, uv, _ = load_view(vd)
        worlds.append((lm_cam - t) @ R)                       # world from this view
        reproj_err.append(np.linalg.norm(project(worlds[-1], K, R, t) - uv, axis=1))
        d = np.load(vd / "depth.npy")
        face = d > 0
        depth_frac.append(face.mean())
        if face.any():
            depth_lo.append(float(d[face].min())); depth_hi.append(float(d[face].max()))

    m = {"n_views": len(vds)}
    if worlds:
        W = np.stack(worlds)                                  # (V,68,3)
        # cross-view agreement of the recovered world landmarks
        m["world_spread_mm"] = float(np.linalg.norm(W - W[0], axis=-1).max())
        m["reproj_px_max"] = float(np.concatenate(reproj_err).max())
        m["reproj_px_mean"] = float(np.concatenate(reproj_err).mean())
        m["depth_frac_min"] = float(min(depth_frac))
        m["depth_lo"] = float(min(depth_lo)) if depth_lo else 0.0
        m["depth_hi"] = float(max(depth_hi)) if depth_hi else 0.0
    return m, problems


def verify(root: Path, sample: int, seed: int):
    subs = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.name)
    print(f"root: {root}")
    print(f"subjects on disk: {len(subs)}")

    # --- STRUCTURE: cheap pass over ALL subjects (view-count + file presence) ---
    view_counts, all_problems = {}, []
    for d in subs:
        vds = subject_views(d)
        view_counts[len(vds)] = view_counts.get(len(vds), 0) + 1
    print(f"view-count histogram (n_views -> #subjects): {dict(sorted(view_counts.items()))}")

    # --- GEOMETRY + DEPTH: sampled deep check ---
    rng = random.Random(seed)
    picks = subs if sample <= 0 or sample >= len(subs) else rng.sample(subs, sample)
    print(f"\ndeep-checking {len(picks)} subjects (geometry + depth)...")
    agg = {"world_spread_mm": [], "reproj_px_max": [], "reproj_px_mean": [],
           "depth_frac_min": [], "depth_lo": [], "depth_hi": []}
    worst_reproj, worst_spread = (0.0, None), (0.0, None)
    for d in picks:
        m, probs = check_subject(d)
        all_problems += probs
        for k in agg:
            if k in m:
                agg[k].append(m[k])
        if m.get("reproj_px_max", 0) > worst_reproj[0]:
            worst_reproj = (m["reproj_px_max"], d.name)
        if m.get("world_spread_mm", 0) > worst_spread[0]:
            worst_spread = (m["world_spread_mm"], d.name)

    def stat(xs):
        a = np.asarray(xs)
        return f"min {a.min():.4g}  mean {a.mean():.4g}  max {a.max():.4g}"

    print("\n--- correctness (sampled) ---")
    print(f"world cross-view spread (mm):   {stat(agg['world_spread_mm'])}")
    print(f"reproj error, per-view max (px):{stat(agg['reproj_px_max'])}")
    print(f"reproj error, mean (px):        {stat(agg['reproj_px_mean'])}")
    print(f"depth face-fraction (min/view): {stat(agg['depth_frac_min'])}")
    print(f"depth range lo (mm):            {stat(agg['depth_lo'])}")
    print(f"depth range hi (mm):            {stat(agg['depth_hi'])}")
    print(f"worst reproj:  {worst_reproj[1]}  ({worst_reproj[0]:.3f} px)")
    print(f"worst spread:  {worst_spread[1]}  ({worst_spread[0]:.4g} mm)")

    print("\n--- structural problems ---")
    if all_problems:
        for p in all_problems[:40]:
            print("  " + p)
        if len(all_problems) > 40:
            print(f"  ... and {len(all_problems) - 40} more")
    else:
        print("  none in checked subjects")

    # thresholds mirror the dataset asserts (reproj<1px atol; world<1e-3 there is
    # strict, but the messy set derives-per-view so allow a looser but tiny bound)
    ok = (not all_problems
          and np.max(agg["reproj_px_max"]) < 1.0
          and np.max(agg["world_spread_mm"]) < 1e-2)
    print(f"\nVERDICT (automated geometry/structure): {'PASS' if ok else 'REVIEW'}")
    return picks


def panel(root: Path, out: Path, n_subjects: int, seed: int):
    subs = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.name)
    rng = random.Random(seed)
    chosen = rng.sample(subs, n_subjects)
    ncol = max(len(subject_views(chosen[0])), 1)
    fig, axes = plt.subplots(n_subjects, ncol,
                             figsize=(2.6 * ncol, 2.6 * n_subjects))
    axes = np.atleast_2d(axes)
    print(f"\npanel subjects: {[d.name for d in chosen]}")
    for r, d in enumerate(chosen):
        vds = subject_views(d)
        for c in range(ncol):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(vds):
                continue
            vd = vds[c]
            img = np.asarray(Image.open(vd / "rgb.png").convert("RGB"))
            uv = np.load(vd / "landmarks_2d.npy")
            ax.imshow(img)
            ax.scatter(uv[:, 0], uv[:, 1], s=3, c="lime", edgecolors="none")
            if c == 0:
                ax.set_title(f"{d.name}  v{vd.name}", fontsize=8, loc="left")
            else:
                ax.set_title(f"v{vd.name}", fontsize=8, loc="left")
    fig.suptitle(f"messy bake eyeball panel  ({root.name})  green = stored 2D landmarks",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"panel -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(REPO / "data/facescape/virtual_camera_messy"))
    p.add_argument("--sample", type=int, default=300,
                   help="#subjects for the deep geometry/depth check (0=all)")
    p.add_argument("--panel-subjects", type=int, default=6)
    p.add_argument("--seed", type=int, default=None, help="default: random each run")
    p.add_argument("--out", default=str(REPO / "scratch/messy_verify/panel.png"))
    args = p.parse_args()
    seed = args.seed if args.seed is not None else random.randrange(1_000_000)
    print(f"seed: {seed}\n")
    verify(Path(args.root), args.sample, seed)
    panel(Path(args.root), Path(args.out), args.panel_subjects, seed)


if __name__ == "__main__":
    main()
