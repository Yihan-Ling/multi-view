import numpy as np
import warnings
import sys
from dataclasses import dataclass
from pathlib import Path
import random
from math import floor
import shutil
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_fill_holes

try:
    from tqdm import tqdm
except ModuleNotFoundError:                            # keep .venv install-free
    def tqdm(iterable, desc="", **kwargs):
        total = kwargs.get("total") or (len(iterable) if hasattr(iterable, "__len__") else None)
        for i, x in enumerate(iterable, 1):
            yield x
            if total and (i % 50 == 0 or i == total):
                end = "\n" if i == total else ""
                print(f"\r{desc}: {i}/{total}", end=end, file=sys.stderr, flush=True)

# Shared RetinaFace crop helper lives next to this script (scripts/facescape/).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from face_detector_crop import detect_main_box, box_to_center_scale  # noqa: E402


@dataclass
class View:
    subject_id: str
    cam_id: str
    light_id: str          # "rgb", "rgb_1", ...
    rgb_path: Path
    landmark_path: Path


# iBUG-68 index groups (0-indexed) -- reference constants for center/scale + the
# inter-ocular normalizer. The logic that uses them is yours.
LM_RIGHT_EYE = list(range(36, 42))   # subject's right eye  (outer corner = 36)
LM_LEFT_EYE  = list(range(42, 48))   # subject's left eye   (outer corner = 45)
LM_OUTER_EYE_CORNERS = (36, 45)      # inter-ocular distance = ||lm[36]-lm[45]||

CAM_IDS = {str(i) for i in range(10)}
LIGHT_VARIANTS = ["rgb.png", "rgb_1.png", "rgb_2.png", "rgb_3.png", "rgb_4.png"]


def discover_views(data_root: Path, lighting: bool = True) -> list[View]:
    views: list[View] = []
    for id_dir in sorted(data_root.iterdir()):
        if not id_dir.is_dir():
            continue
        subject_id = id_dir.name

        for cam_dir in sorted(id_dir.iterdir()):
            if not cam_dir.is_dir() or cam_dir.name not in CAM_IDS:
                continue
            cam_id = cam_dir.name

            landmark_path = cam_dir / "landmarks_2d.npy"
            if not landmark_path.exists():
                warnings.warn(f"landmark file not found: {landmark_path}")
                continue

            for rgb in (LIGHT_VARIANTS if lighting else ["rgb.png"]):
                rgb_path = cam_dir / rgb
                if not rgb_path.exists():
                    warnings.warn(f"rgb image not found: {rgb_path}")
                    continue
                views.append(View(subject_id, cam_id, Path(rgb).stem,
                                  rgb_path, landmark_path))
    return views



def split_val(views: list[View], rng: np.random.Generator, val_fraction = 0.2):
    subject_ids = set()
    for v in views:
        subject_ids.add(v.subject_id)
    subject_ids = sorted(subject_ids)
    
    rng.shuffle(subject_ids)
    n_val = max(floor(val_fraction*len(subject_ids)), 1)
    val_ids = set(subject_ids[0:n_val])
    
    train, val = [], []
    for v in views:
        if v.subject_id in val_ids:
            val.append(v)
        else:
            train.append(v)
    
    return {"train": train, "val": val}



def view2row(view: View):
    IMG_W, IMG_H = 512, 512 
    landmarks = np.load(view.landmark_path)
    in_img = (landmarks[:,0] >=0) & (landmarks[:,0]<IMG_W) & (landmarks[:,1] >= 0) & (landmarks[:,1] < IMG_H)
    
    visable = landmarks[in_img]
    
    x0, y0 = visable.min(axis=0)
    x1, y1 = visable.max(axis=0)
    w = x1 - x0
    h = y1 - y0
    
    PAD = 1.25     # our landmark boxes are tight (no forehead/ears); pad so the whole head fits
    NUDGE = 0.08   # gentle upward bias for the missing forehead

    center_w = (x0+x1)/2
    center_h = (y0 + y1) / 2 - h * NUDGE
    scale    = (w + h) / 2 / 200 * PAD
    
    out  = landmarks.copy()
    out[~in_img]  = -1.0   
    flat_pts = out.flatten()
    relpath = f"{view.subject_id}_{view.cam_id}_{view.light_id}.png"
    
    return (relpath, scale, center_w, center_h, flat_pts)


def load_background_paths(bg_root: Path) -> list[Path]:
    paths = []
    for ext in ("*.jpg", "*.png"):
        paths.extend(bg_root.rglob(ext))
    return sorted(paths)


def _fit_bg(bg: np.ndarray, H: int, W: int, rng) -> np.ndarray:
    Hb, Wb = bg.shape[:2]
    s = max(H / Hb, W / Wb)
    new_w, new_h = max(W, round(Wb * s)), max(H, round(Hb * s))
    bg = np.asarray(Image.fromarray(bg).resize((new_w, new_h), Image.BILINEAR))
    y0 = int(rng.integers(0, new_h - H + 1))
    x0 = int(rng.integers(0, new_w - W + 1))
    return bg[y0:y0 + H, x0:x0 + W]


def fill_interior_holes(img: np.ndarray, mask: np.ndarray, fill_value: int) -> np.ndarray:
    # The TU mesh has no eyeballs, so the eye region is a depth==0 HOLE. On a
    # composited image the random background would leak THROUGH the eyes; on a
    # clean render the render's own (black) background shows through them. Either
    # way the eyes must be a consistent color. binary_fill_holes gives the solid
    # head silhouette; depth==0 pixels enclosed by it are interior holes (eyes,
    # nostrils) -> paint them a constant `fill_value`. Mutates `img` in place.
    solid = binary_fill_holes(mask)
    interior_holes = solid & ~mask
    img[interior_holes] = fill_value
    return img


def composite_over_bg(rgb: np.ndarray, depth: np.ndarray, bg: np.ndarray,
                      rng, fill_holes: bool = True, fill_value: int = 0) -> np.ndarray:
    H, W = rgb.shape[:2]
    mask = depth > 0                                   # rendered face surface
    bg = _fit_bg(bg, H, W, rng).astype(np.uint8)
    out = np.where(mask[..., None], rgb, bg).astype(np.uint8)
    if fill_holes:
        out = fill_interior_holes(out, mask, fill_value)
    return out


def build_csv(views: list[View], train_type, out_root: Path,
              bg_paths: list[Path], rng: np.random.Generator, composite_prob = 0.8,
              fill_holes: bool = True, fill_value: int = 0, crop_mode: str = "landmark",
              det_gpu_id: int = -1, det_network: str = "mobilenet"):
    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    n_fallback = 0
    for v in tqdm(views, desc=f"build {train_type}", unit="view"):
        relpath, lm_scale, lm_cw, lm_ch, flat_pts = view2row(v)
        out_path = images_dir / relpath

        # Produce the FINAL on-disk image; `img` (RGB array) is kept when the
        # detector needs to run on those exact pixels (retinaface crop mode).
        if rng.random() < composite_prob:
            rgb = np.asarray(Image.open(v.rgb_path).convert("RGB"))
            depth = np.load(v.rgb_path.parent / "depth.npy")
            background = np.asarray(Image.open(bg_paths[rng.integers(len(bg_paths))]).convert("RGB"))
            img = composite_over_bg(rgb, depth, background, rng,
                                    fill_holes=fill_holes, fill_value=fill_value)
            Image.fromarray(img).save(out_path)
        elif fill_holes:
            # No background composited, but still paint the eye/nostril holes so the
            # eyes match the composited images instead of showing the render's black bg.
            rgb = np.asarray(Image.open(v.rgb_path).convert("RGB"))
            depth = np.load(v.rgb_path.parent / "depth.npy")
            img = fill_interior_holes(rgb.copy(), depth > 0, fill_value)
            Image.fromarray(img).save(out_path)
        else:
            shutil.copy(v.rgb_path, out_path)
            img = None

        # Framing: detector box (parity with real deployment) or landmark bbox.
        if crop_mode == "retinaface":
            if img is None:
                img = np.asarray(Image.open(out_path).convert("RGB"))
            box = detect_main_box(img, gpu_id=det_gpu_id, network=det_network)
            if box is not None:
                scale, center_w, center_h = box_to_center_scale(box)
            else:                                   # no face detected -> fall back
                scale, center_w, center_h = lm_scale, lm_cw, lm_ch
                n_fallback += 1
        else:
            scale, center_w, center_h = lm_scale, lm_cw, lm_ch

        rows.append([relpath, scale, center_w, center_h, *flat_pts])

    if crop_mode == "retinaface":
        print(f"[{train_type}] retinaface: {n_fallback}/{len(views)} views had no "
              f"detection -> fell back to landmark bbox")

    cols = ["image", "scale", "center_w", "center_h"] + [f"p{i}" for i in range(136)]

    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_root/f"{train_type}.csv", index=False)


def main(data_root, out_root,
         bg_root="data/backgrounds/indoor/Images", seed=0, composite_prob=0.8,
         fill_holes=True, fill_value=0, crop_mode="landmark", det_gpu_id=-1,
         det_network="mobilenet"):

    rng = np.random.default_rng(seed)

    data_root = Path(data_root)
    out_root  = Path(out_root)
    views = discover_views(data_root)
    split = split_val(views, rng=rng)

    # Skip loading the background pool when compositing is disabled, so a
    # photometric-only build doesn't require the Indoor67 dir to be present.
    bg_paths = load_background_paths(Path(bg_root)) if composite_prob > 0 else []
    for n in ("train", "val"):
        build_csv(split[n], n, out_root, bg_paths, rng, composite_prob,
                  fill_holes, fill_value, crop_mode=crop_mode, det_gpu_id=det_gpu_id,
                  det_network=det_network)


def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Build the HRNet 68-pt landmark bundle from FaceScape virtual-camera renders.")
    p.add_argument("--data-root", default="data/facescape/virtual_camera_data",
                   help="Root of the virtual-camera renders.")
    p.add_argument("--out-root", default="data/facescape/HRNet_train_nobg",
                   help="Output bundle dir (images/ + train.csv/val.csv).")
    p.add_argument("--bg-root", default="data/backgrounds/indoor/Images",
                   help="Background pool (ignored when --composite-prob is 0).")
    p.add_argument("--composite-prob", type=float, default=0.0,
                   help="Per-view probability of compositing a random background "
                        "(0 = clean renders / no background aug).")
    p.add_argument("--fill-holes", action="store_true", default=True,
                   help="Paint interior depth==0 holes (eyes) a constant color instead "
                        "of letting the background leak through them (default on). "
                        "Color set by --fill-color.")
    p.add_argument("--no-fill-holes", dest="fill_holes", action="store_false",
                   help="Disable hole filling; background leaks through the eye holes.")
    p.add_argument("--fill-color", choices=["black", "white"], default="black",
                   help="Color for filled eye/nostril holes (only with --fill-holes): "
                        "'black' = 0 (old default), 'white' = 255.")
    p.add_argument("--crop-mode", choices=["landmark", "retinaface"], default="landmark",
                   help="Framing source: 'landmark' = GT-landmark bbox (old default); "
                        "'retinaface' = RetinaFace detection box (must match the eval "
                        "converters' --crop-mode for train/test parity).")
    p.add_argument("--det-gpu", type=int, default=-1,
                   help="GPU id for the RetinaFace detector (-1 = CPU). Only used with "
                        "--crop-mode retinaface.")
    p.add_argument("--det-network", choices=["mobilenet", "resnet50"], default="mobilenet",
                   help="RetinaFace backbone for detection (mobilenet = fast/small).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.data_root, args.out_root,
         bg_root=args.bg_root, seed=args.seed, composite_prob=args.composite_prob,
         fill_holes=args.fill_holes, fill_value=(255 if args.fill_color == "white" else 0),
         crop_mode=args.crop_mode,
         det_gpu_id=args.det_gpu, det_network=args.det_network)