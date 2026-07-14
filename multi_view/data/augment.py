"""Domain-randomization augmentation for the iteration-2 multi-view robustness
study -- the SAME sim-to-real recipe that worked for the HRNet landmark model.

Two on-the-fly, per-view augmentations applied INSIDE the Dataset on the raw RGB
(float [0,1], HxWx3) BEFORE depth normalization, so the depth>0 face mask is
still available:
  - background compositing: replace the rendered (empty/black) background with a
    random indoor photo, keyed off the raw depth>0 face mask. Because that mask
    excludes the eyeless eye holes (interior depth==0 regions), the background
    deliberately bleeds THROUGH the eyes here -- the opposite of the HRNet eye-leak
    fix, requested for this multi-view run.
  - photometric jitter: the EXACT HRNet `photometric()` pipeline (brightness /
    contrast / saturation / hue / blur / downscale / additive sensor noise / JPEG).
    Reused verbatim from scripts/facescape/hrnet/facescape_aug.py so the two tracks
    never diverge -- NOT a blur-only stand-in. Order matches HRNet: composite the
    background first, then run the whole image through the camera-ISP pipeline (the
    face and its new background go through one sensor together).

Only RGB is augmented; the depth channel and all GT labels (landmarks_3d/2d, vis)
are left untouched, so MPJPE stays an honest metric. Determinism is controlled by
the caller-supplied numpy Generator: a per-sample seeded RNG on the val split gives
a stable metric; a fresh-entropy RNG on train gives fresh augmentation each epoch.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# Reuse HRNet's photometric pipeline verbatim (parity with the landmark track).
# facescape_aug.py guards its HRNet-lib import, so photometric() imports standalone.
_HRNET_DIR = Path(__file__).resolve().parents[2] / "scripts/facescape/hrnet"
if str(_HRNET_DIR) not in sys.path:
    sys.path.insert(0, str(_HRNET_DIR))
from facescape_aug import photometric  # noqa: E402


@dataclass
class AugConfig:
    bg_dir: str | None = None      # directory of background images (recursively globbed)
    bg_prob: float = 0.0           # per-view probability of compositing a background
    photometric: bool = False      # apply the HRNet photometric ISP pipeline per view

    @property
    def enabled(self) -> bool:
        return (self.bg_prob > 0 and bool(self.bg_dir)) or self.photometric


class MultiViewAugmentor:
    """Holds the background pool; applies bg composite + HRNet photometric jitter
    to one view's RGB."""

    def __init__(self, cfg: AugConfig):
        self.cfg = cfg
        self.bg_paths: list[Path] = []
        if cfg.bg_dir and cfg.bg_prob > 0:
            root = Path(cfg.bg_dir)
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                self.bg_paths.extend(sorted(root.rglob(ext)))
            if not self.bg_paths:
                raise FileNotFoundError(f"no background images found under {root}")

    def _fit_bg(self, bg: np.ndarray, H: int, W: int, rng) -> np.ndarray:
        """Scale-to-cover then random-crop the background to (H, W)."""
        Hb, Wb = bg.shape[:2]
        s = max(H / Hb, W / Wb)
        new_w, new_h = max(W, round(Wb * s)), max(H, round(Hb * s))
        bg = np.asarray(Image.fromarray(bg).resize((new_w, new_h), Image.BILINEAR))
        y0 = int(rng.integers(0, new_h - H + 1))
        x0 = int(rng.integers(0, new_w - W + 1))
        return bg[y0:y0 + H, x0:x0 + W]

    def apply(self, rgb: np.ndarray, face_mask: np.ndarray, rng) -> np.ndarray:
        """rgb (H,W,3) float [0,1]; face_mask (H,W) bool raw depth>0 face pixels
        (eye holes are False, so bg shows through the eyes). Returns the augmented
        rgb (H,W,3) float [0,1]."""
        H, W = rgb.shape[:2]
        # Background composite: keep only the rendered face pixels, swap everything
        # else -- including the interior eye holes -- for a random indoor photo.
        if self.bg_paths and rng.random() < self.cfg.bg_prob:
            bp = self.bg_paths[int(rng.integers(len(self.bg_paths)))]
            bg = np.asarray(Image.open(bp).convert("RGB"), dtype=np.uint8)
            bg = self._fit_bg(bg, H, W, rng).astype(np.float32) / 255.0
            rgb = np.where(face_mask[..., None], rgb, bg).astype(np.float32)
        # HRNet photometric ISP pipeline on the whole (composited) RGB. photometric()
        # works in uint8; each internal effect fires on its own probability off `rng`,
        # so a seeded rng makes the val augmentation deterministic.
        if self.cfg.photometric:
            u8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
            rgb = photometric(u8, rng).astype(np.float32) / 255.0
        return rgb
