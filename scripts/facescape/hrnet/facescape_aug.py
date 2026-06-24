from __future__ import annotations

import io
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

# The HRNet vendored lib (Face300W base + transform helpers) is only needed for
# TRAINING. Guard the import so the pure-numpy photometric() function below can be
# imported standalone (e.g. by the viz tool) without the HRNet lib on the path.
# During training PYTHONPATH points at the HRNet lib, so this import succeeds.
try:
    from lib.datasets.face300w import Face300W
    from lib.utils.transforms import fliplr_joints, crop, generate_target, transform_pixel
except ImportError:  # standalone use (visualization / checking the aug menu)
    Face300W = object
    fliplr_joints = crop = generate_target = transform_pixel = None


# Per-effect probabilities + strength ranges, weighted by the 2026-06-22 aug-probe:
# additive sensor NOISE was the dominant real-image fragility (std=8 -> NME 1.38),
# then desaturation / contrast / JPEG; blur and downscale were comparatively benign.
# So noise + color fire often and hard; blur/downscale stay gentle and rare.
def photometric(img: np.ndarray, rng: np.random.Generator, log: dict | None = None) -> np.ndarray:
    """Photometrically jitter a uint8 HxWx3 RGB image; return uint8 HxWx3.

    Pure pixel ops only -- nothing here moves a pixel geometrically, so the GT
    landmarks stay valid and are not passed in. A random SUBSET fires each call
    (the model must see clean and degraded variants). Order mimics a camera
    pipeline: appearance/ISP -> optics -> resolution -> sensor noise -> JPEG.

    `log` is for verification only (see verify_aug.py): pass a dict and each key
    gets the sampled strength when its effect fired, else None. Default None = a
    pure no-op, so the training path is byte-for-byte unchanged.
    """
    def rec(name, value):  # record the sampled strength (or None if it didn't fire)
        if log is not None:
            log[name] = value

    pil = Image.fromarray(img)

    # --- appearance / ISP: skin tone + white balance + exposure ---
    fired = rng.random() < 0.5
    f = rng.uniform(0.6, 1.4) if fired else None
    if fired:
        pil = ImageEnhance.Brightness(pil).enhance(f)
    rec('brightness', f)

    fired = rng.random() < 0.7
    f = rng.uniform(0.4, 1.6) if fired else None      # spans probe 0.35 break
    if fired:
        pil = ImageEnhance.Contrast(pil).enhance(f)
    rec('contrast', f)

    fired = rng.random() < 0.7
    f = rng.uniform(0.4, 1.6) if fired else None      # saturation; spans 0.75 break
    if fired:
        pil = ImageEnhance.Color(pil).enhance(f)
    rec('saturation', f)

    fired = rng.random() < 0.3
    s = int(rng.integers(-15, 16)) if fired else None
    if fired:
        hsv = np.asarray(pil.convert('HSV'), dtype=np.int16)
        hsv[..., 0] = (hsv[..., 0] + s) % 256  # hue wraps
        pil = Image.fromarray(hsv.astype(np.uint8), 'HSV').convert('RGB')
    rec('hue', s)

    # --- optics: soft focus / cheap lens (minor per the probe) ---
    fired = rng.random() < 0.3
    r = rng.uniform(0.0, 2.5) if fired else None
    if fired:
        pil = pil.filter(ImageFilter.GaussianBlur(r))
    rec('blur', r)

    # --- resolution: low-res capture then upsample (minor) ---
    fired = rng.random() < 0.2
    f = rng.uniform(0.4, 1.0) if fired else None
    if fired:
        w, h = pil.size
        pil = pil.resize((max(1, int(w * f)), max(1, int(h * f))), Image.BILINEAR)
        pil = pil.resize((w, h), Image.BILINEAR)
    rec('downscale', f)

    arr = np.asarray(pil, dtype=np.float32)

    # --- sensor: additive gaussian noise (THE big lever; fires most often) ---
    fired = rng.random() < 0.85
    sigma = rng.uniform(0.0, 16.0) if fired else None  # spans probe 8 break
    if fired:
        arr = arr + rng.normal(0.0, sigma, arr.shape)
    rec('noise', sigma)
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    # --- compression: JPEG re-encode (real photos are JPEG; renders are not) ---
    fired = rng.random() < 0.5
    q = int(rng.integers(20, 91)) if fired else None   # spans probe q=12
    if fired:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format='JPEG', quality=q)
        buf.seek(0)
        arr = np.asarray(Image.open(buf).convert('RGB'), dtype=np.uint8)
    rec('jpeg', q)

    return arr


class FaceScapeAug(Face300W):
    def __init__(self, cfg, is_train=True, transform=None):
        super().__init__(cfg, is_train=is_train, transform=transform)
        # Wave 2 only: pool of background images + how often to swap.
        # Leave these unused until Wave 2; Wave 1 ignores them.
        # self.bg_paths = ...            # TODO(you): load_background_paths(cfg...)
        # self.composite_prob = ...      # e.g. 0.8, mirror the adapter
        self.rng = np.random.default_rng()  # cheap per-worker rng for bg choice

    # ---- Wave 1: photometric realism ---------------------------------------
    def _photometric(self, img: np.ndarray) -> np.ndarray:
        return photometric(img, self.rng)

    # ---- Wave 2: live background swap (enable after the adapter change) -----
    def _composite_bg(self, img: np.ndarray, idx: int) -> np.ndarray:
        """Paste a random background behind the head using the depth>0 matte.

        Reuse composite_over_bg() from build_hrnet_landmark_dataset.py (don't
        re-derive the matte math). Needs, for this idx: the depth map for the
        same view + a randomly chosen bg from self.bg_paths. Roll a random number
        against self.composite_prob so some fraction keep the clean render.
        Requires the Wave-2 bundle (clean img + depth shipped). No-op for now.
        """
        # TODO(you, Wave 2): load depth for this idx, pick a bg, composite.
        return img

    # ---- inherited plumbing with the two aug hooks inserted ----------------
    # NOTE: this body is copied from Face300W.__getitem__ (vendored, stable). The
    # ONLY additions are the two augmentation calls right after the image load,
    # flagged with  # <-- AUG. Everything else is the parent's exact sequence;
    # do not retune it (parity with how the model scores -- see eval_real).
    def __getitem__(self, idx):
        image_path = f"{self.data_root}/{self.landmarks_frame.iloc[idx, 0]}"
        scale = self.landmarks_frame.iloc[idx, 1]
        center_w = self.landmarks_frame.iloc[idx, 2]
        center_h = self.landmarks_frame.iloc[idx, 3]
        center = torch.Tensor([center_w, center_h])

        pts = self.landmarks_frame.iloc[idx, 4:].values.astype('float').reshape(-1, 2)
        scale *= 1.25
        nparts = pts.shape[0]
        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.uint8)

        if self.is_train:
            img = self._composite_bg(img, idx)   # <-- AUG (Wave 2; no-op for now)
            img = self._photometric(img)         # <-- AUG (Wave 1)

        img = img.astype(np.float32)

        r = 0
        if self.is_train:
            scale = scale * random.uniform(1 - self.scale_factor, 1 + self.scale_factor)
            r = random.uniform(-self.rot_factor, self.rot_factor) if random.random() <= 0.6 else 0
            if random.random() <= 0.5 and self.flip:
                img = np.fliplr(img)
                pts = fliplr_joints(pts, width=img.shape[1], dataset='300W')
                center[0] = img.shape[1] - center[0]

        img = crop(img, center, scale, self.input_size, rot=r)

        target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        tpts = pts.copy()
        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2] + 1, center, scale,
                                               self.output_size, rot=r)
                target[i] = generate_target(target[i], tpts[i] - 1, self.sigma,
                                            label_type=self.label_type)

        img = (img / 255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])

        meta = {'index': idx, 'center': center, 'scale': scale,
                'pts': torch.Tensor(pts), 'tpts': torch.Tensor(tpts)}
        return img, torch.Tensor(target), meta


# =============================================================================
# NOTE -- Wave 2 adapter change (do later, not now):
#   build_hrnet_landmark_dataset.py currently SAVES the pre-composited image and
#   ships no depth. For live bg, the bundle must instead carry, per view:
#     - the CLEAN render (images/<name>.png)
#     - the depth matte (e.g. images/<name>_depth.npy or a packed mask)
#   and the MIT Indoor67 pool must be present on the desktop. Then enable
#   self.bg_paths / self.composite_prob in __init__ and fill _composite_bg().
# =============================================================================
