"""Convert rendered FaceScape views into an HRNet facial-landmark training set.

Sim-to-real pivot (2026-06-16): we train HRNetV2-W18 on these synthetic renders
to regress 68-pt 2D landmarks (RGB-only), then test on REAL AFLW2000-3D + WFLW.
HRNet is a single-image landmark regressor, so the multi-view structure is
irrelevant here: every rendered view becomes ONE training row with 68 points.

------------------------------------------------------------------------------
INPUT  (one folder per view, written by render_views.py)
------------------------------------------------------------------------------
data/facescape/virtual_camera_data/<id>/<cam>/
    rgb.png            HxW RGB render            (the training image)
    landmarks_2d.npy   (68, 2) float64, PIXELS   (u, v) in the rgb.png frame
    meta.json          has W, H, K, R, t, ...    (W,H = image size)
  Lighting variants rgb_1..rgb_4.png share the SAME landmarks_2d (geometry is
  rendered once, then relit), so each variant is an extra training row.
Note: <cam> folders are named 0,1,2,...; skip non-cam entries (panel.png, etc.).

------------------------------------------------------------------------------
OUTPUT  (HRNet facial-landmark format -- VERIFIED against the actual loader
         third_party/HRNet-Facial-Landmark-Detection/lib/datasets/face300w.py)
------------------------------------------------------------------------------
A pandas-readable CSV per split (train.csv / val.csv). The loader does:

    landmarks_frame = pd.read_csv(csv_file)          # FIRST ROW IS A HEADER
    image_path = os.path.join(DATASET.ROOT, frame.iloc[idx, 0])
    scale      = frame.iloc[idx, 1]                   # then internally *= 1.25
    center_w   = frame.iloc[idx, 2]
    center_h   = frame.iloc[idx, 3]
    pts        = frame.iloc[idx, 4:].values.reshape(-1, 2)   # x1,y1,x2,y2,...

So each row is exactly:
    <relpath>, <scale>, <center_w>, <center_h>, x1, y1, x2, y2, ..., x68, y68
  -> 4 + 68*2 = 140 columns, and a header row MUST exist (else pandas eats the
     first data row as column names).

KEY FORMAT FACTS pulled straight from face300w.py (do not guess these):
  * scale is a 200-px-normalized box size: the loader crops a region of
    ~ (scale * 1.25 * 200) px around `center`. The classic 300W convention is
        scale = box_side / 200      (box_side from the landmark extent, see #3)
    Because the loader already multiplies by 1.25, do NOT pre-bake that 1.25.
  * `center` is the crop center in PIXELS (full-image coords), float ok.
  * VALIDITY: the loader skips any point with `pts[i, 1] <= 0` (y <= 0). All our
    rendered landmarks are visible and > 0, so this is fine -- but it means the
    crop/center must keep them positive; clamp center, never the points.
  * INDEXING QUIRK: the loader computes transform_pixel(pts+1) and
    generate_target(pts-1). The stock 300W CSVs hold 1-INDEXED (MATLAB) pixels.
    Our landmarks_2d.npy are 0-indexed numpy pixels. See VERIFY #2 -- decide
    whether to write `lm` or `lm + 1`, and be consistent with how NME is scored.

DATASET.ROOT: pick one root and write relpaths beneath it, so the loader's
os.path.join(ROOT, relpath) resolves. Mirror the train config you'll use:
    third_party/.../experiments/300w/face_alignment_300w_hrnet_w18.yaml
(point its DATASET.ROOT / TRAINSET / TESTSET at what this script emits).

------------------------------------------------------------------------------
TWO THINGS TO VERIFY before trusting the output (do NOT skip):
------------------------------------------------------------------------------
 1. 68-pt ORDER. HRNet/300W expects iBUG-68 order. FaceScape landmarks_2d are
    derived from the toolkit's landmark indices -- confirm they ARE iBUG-68 and
    not some permutation, since flip-augmentation (fliplr_joints(dataset='300W'))
    hard-codes the iBUG-68 left/right mirror pairs. Cheap check: overlay the 68
    pts on rgb.png with their indices and eyeball the canonical iBUG layout
    (jaw 0-16, brows 17-26, nose 27-35, eyes 36-47, mouth 48-67).
 2. 0- vs 1-indexed pixels + the inter-ocular NME convention you'll report,
    so train-time targets and eval-time error use the same pixel origin.
"""

import numpy as np
import warnings
from dataclasses import dataclass
from pathlib import Path
import random
from math import floor
import shutil
import pandas as pd


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


# 1. DISCOVER VIEWS  (shared plumbing, same as the retinaface adapter)
#    Walk <data_root>/<id>/<cam>/, keep cam folders that have landmarks_2d.npy
#    plus at least one rgb variant. Each lighting variant -> its own View row.
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


# ------------------------------------------------------------------------------
# OUTLINE -- you write the functions/signatures/CLI; this is just WHAT it must do.
# ------------------------------------------------------------------------------
#
# 2. SUBJECT-DISJOINT SPLIT
#    Partition the SUBJECT ids (not the views) into train / val so no subject
#    leaks across the split. Group the discovered views by view.subject_id, then
#    hold out a fraction (or an explicit id list) of subjects for val.
#    -> returns {"train": [View, ...], "val": [View, ...]}.

def split_val(views: list[View], val_fraction = 0.2, seed = 0):
    subject_ids = set()
    for v in views:
        subject_ids.add(v.subject_id)
    subject_ids = sorted(subject_ids)
    
    rng = random.Random(seed)
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


# 3. PER VIEW: derive the HRNet row fields from landmarks_2d
#    a. lm = np.load(landmark_path)            # (68, 2) pixel coords
#    b. tight box from the landmark extent:
#         x0,y0 = lm.min(axis=0);  x1,y1 = lm.max(axis=0)
#       NOTE: iBUG-68 stops at the eyebrows -> no forehead. 300W's `scale`
#       expects the FULL face box, so the side wants padding (esp. upward) before
#       you normalize. Decide the box_side (max of w,h? mean? padded how much?)
#       and document it -- this constant directly sets the crop the net sees.
#    c. center = box center in pixels  (you may nudge center_h up for forehead).
#       scale  = box_side / 200.0        # NOT *1.25 -- the loader adds that.
#    d. keep all 68 pts (flattened x,y order); remember VERIFY #2 (0- vs 1-index).
#    -> returns (relpath, scale, center_w, center_h, flat_pts[136]).

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

# 4. PLACE THE IMAGE
#    Copy or symlink rgb.png to <out_root>/<DATASET.ROOT>/images/<relpath>, with
#    a relpath unique per view, e.g. f"{id}_{cam}_{light}.png". The CSV's col-0
#    relpath must be what os.path.join(DATASET.ROOT, relpath) resolves to.

def build_csv(views: list[View], train_type, out_root: Path):
    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for v in views:
        relpath, scale, center_w, center_h, flat_pts = view2row(v)
        
        shutil.copy(v.rgb_path, images_dir/relpath)
        
        row= [relpath, scale, center_w, center_h, *flat_pts]
        rows.append(row)
        
    cols = ["image", "scale", "center_w", "center_h"] + [f"p{i}" for i in range(136)]
    
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_root/f"{train_type}.csv", index=False)


# 6. CLI (write last, your choice of args): data root, out root, val fraction or
#    val ids, box padding / forehead nudge, copy-vs-symlink, include-lighting.

def main(data_root, out_root):
    data_root = Path(data_root)
    out_root  = Path(out_root)
    views = discover_views(data_root)
    split = split_val(views)
    
    for n in ("train", "val"):
        build_csv(split[n], n, out_root)
        

if __name__ == "__main__":
    main("data/facescape/virtual_camera_data", "data/facescape/HRNet_train")