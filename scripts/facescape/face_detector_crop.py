"""RetinaFace-based face cropping, shared by the training-bundle builder
(build_hrnet_landmark_dataset.py) and the two real-image eval converters
(build_aflw2000_csv.py, build_wflw_csv.py).

The point of the RetinaFace-crop iteration: train and test must see the SAME
detector-derived framing. Both landmark-bbox framing (the old default) and this
detector framing stay available; the iteration flips every builder to this one so
the model trains and is evaluated on identically-framed crops -- the framing that a
real deployment (detect -> crop -> landmark) would actually produce.

Detector = batch-face RetinaFace (mobilenet by default; weights auto-download on
first use). A module-level singleton is built lazily so importing this module is
cheap and the detector is created at most once per process.

CSV framing convention (must match build_hrnet_landmark_dataset.py):
    scale     = max(box_w, box_h) / 200 * DET_PAD
    center    = box center, optionally nudged down by NUDGE * box_h
The HRNet Face300W loader multiplies `scale` by 1.25 internally, so the final crop
side ~= 1.25 * DET_PAD * max(box_w, box_h). DET_PAD is the one eyeball knob: it sets
how much context around the detected box the crop keeps. Tune it once with
viz_detector_crop.py so synthetic + real crops look alike, then leave it fixed.
"""
from __future__ import annotations

import numpy as np

# --- Framing knobs (eyeball-tuned; keep identical across train + both eval sets) --
DET_PAD = 1.15   # pad on the square-ified detector box before /200 (see module docstring)
NUDGE = 0.0      # vertical center bias as a fraction of box height (down = +). RetinaFace
                 # boxes already include brow->chin, so no forehead nudge is needed.

_detector = None
_detector_key = None


def get_detector(gpu_id: int = -1, network: str = "mobilenet"):
    """Lazily build (and cache) a single RetinaFace detector. gpu_id=-1 -> CPU."""
    global _detector, _detector_key
    key = (gpu_id, network)
    if _detector is None or _detector_key != key:
        from batch_face import RetinaFace
        _detector = RetinaFace(gpu_id=gpu_id, network=network)
        _detector_key = key
    return _detector


def detect_main_box(img_rgb: np.ndarray, threshold: float = 0.5,
                    gpu_id: int = -1, network: str = "mobilenet"):
    """Return (x0, y0, x1, y1) floats for the most confident face, or None.

    img_rgb: HxWx3 uint8 RGB array (cv=False tells batch-face the input is RGB).
    """
    det = get_detector(gpu_id=gpu_id, network=network)
    faces = det(img_rgb, threshold=threshold, cv=False)
    if not faces:
        return None
    box, _kps, _score = max(faces, key=lambda f: f[2])   # highest confidence
    x0, y0, x1, y1 = (float(v) for v in box)
    return x0, y0, x1, y1


def box_to_center_scale(box, pad: float = DET_PAD, nudge: float = NUDGE):
    """(x0,y0,x1,y1) -> (scale, center_w, center_h) in the HRNet CSV convention."""
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    center_w = (x0 + x1) / 2.0
    center_h = (y0 + y1) / 2.0 + h * nudge
    scale = max(w, h) / 200.0 * pad
    return scale, center_w, center_h
