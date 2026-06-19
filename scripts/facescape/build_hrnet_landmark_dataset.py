import numpy as np
import warnings
from dataclasses import dataclass
from pathlib import Path
import random
from math import floor
import shutil
import pandas as pd
from PIL import Image


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


def composite_over_bg(rgb: np.ndarray, depth: np.ndarray, bg: np.ndarray,
                      rng) -> np.ndarray:
    H, W = rgb.shape[:2]
    mask = depth > 0
    bg = _fit_bg(bg, H, W, rng).astype(np.uint8)
    return np.where(mask[..., None], rgb, bg).astype(np.uint8)


def build_csv(views: list[View], train_type, out_root: Path,
              bg_paths: list[Path], rng: np.random.Generator, composite_prob = 0.8):
    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for v in views:
        relpath, scale, center_w, center_h, flat_pts = view2row(v)
        out_path = images_dir / relpath

        
        if rng.random() < composite_prob:
            rgb = np.asarray(Image.open(v.rgb_path).convert("RGB"))
            depth = np.load(v.rgb_path.parent / "depth.npy")
            background = np.asarray(Image.open(bg_paths[rng.integers(len(bg_paths))]).convert("RGB"))
            
            Image.fromarray(composite_over_bg(rgb, depth, background, rng)).save(out_path)
        
        else:
            shutil.copy(v.rgb_path, out_path)

        row= [relpath, scale, center_w, center_h, *flat_pts]
        rows.append(row)
        
    cols = ["image", "scale", "center_w", "center_h"] + [f"p{i}" for i in range(136)]
    
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_root/f"{train_type}.csv", index=False)


def main(data_root, out_root,
         bg_root="data/backgrounds/indoor/Images", seed=0, composite_prob=0.8):
    
    rng = np.random.default_rng(seed)
    
    data_root = Path(data_root)
    out_root  = Path(out_root)
    views = discover_views(data_root)
    split = split_val(views, rng=rng)
    
    bg_paths = load_background_paths(Path(bg_root))
    for n in ("train", "val"):
        build_csv(split[n], n, out_root, bg_paths, rng, composite_prob)


if __name__ == "__main__":
    main("data/facescape/virtual_camera_data", "data/facescape/HRNet_train")