"""Multi-view RGBD dataset for the early-fusion 3D-landmark pipeline.

One item = one face (one subject) seen from all N virtual cameras, packaged for
the MVGFormer-style decoder:

    rgbd          (N, 4, H, W)  RGB in [0,1] + normalized depth as the 4th channel
    proj          (N, 3, 4)     projection matrix P = K @ [R|t] per view
    landmarks_3d  (68, 3)       GT in the WORLD frame (shared across views)
    landmarks_2d  (N, 68, 2)    GT pixel landmarks per view
    vis           (N, 68)       per-view visibility (geometric occlusion test)

Two on-disk schemas are supported (auto-detected by the presence of a
subject-level ``landmarks_world.npy``):

NEW (scripts/facescape/render_views.py):
    <subj>/landmarks_world.npy          (68,3) world, shared across views
    <subj>/<view>/rgb.png
    <subj>/<view>/depth.npy             (H,W), 0 = hole/background
    <subj>/<view>/landmarks_cam.npy     (68,3) camera frame
    <subj>/<view>/meta.json             K, Rt, P, width, height, landmark_uv, ...

OLD (data/facescape/virtual_camera_data, the iter-1 training set):
    <subj>/<view>/rgb.png
    <subj>/<view>/depth.npy             (H,W), 0 = hole/background
    <subj>/<view>/landmarks_3d.npy      (68,3) CAMERA frame  (== new's landmarks_cam)
    <subj>/<view>/landmarks_2d.npy      (68,2) pixels        (== new's landmark_uv)
    <subj>/<view>/meta.json             K, R, t, W, H, head_quat, ...  (no Rt/P)
    (world landmarks are NOT stored; we derive them: world = (lm_cam - t) @ R,
     which agrees across all views to ~1e-13.)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_fill_holes
from torch.utils.data import Dataset


def subject_disjoint_split(subject_ids, val_frac: float = 0.2, seed: int = 0):
    """Partition subject ids into (train, val) with NO identity in both. Mirrors
    the split logic in scripts/facescape/build_hrnet_landmark_dataset.py."""
    ids = sorted(subject_ids)
    random.Random(seed).shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac)))
    return ids[n_val:], ids[:n_val]


def discover_subjects(root):
    root = Path(root)
    return sorted((d.name for d in root.iterdir() if d.is_dir() and d.name.isdigit()),
                  key=int)


def _project_np(pts_world: np.ndarray, P: np.ndarray) -> np.ndarray:
    """(68,3) world -> (68,2) pixels through P (3,4). numpy mirror of geometry.project."""
    ph = np.concatenate([pts_world, np.ones((pts_world.shape[0], 1))], axis=1)  # (68,4)
    uvw = ph @ P.T                                                              # (68,3)
    return uvw[:, :2] / uvw[:, 2:3]


class MultiViewFaceScape(Dataset):
    def __init__(
        self,
        root: str | Path,
        subject_ids: list[str],
        depth_scale: float = 200.0,
        vis_tol: float = 10.0,
        transform=None,
    ):
        # subject-disjoint split lives at the caller: pass the subject id list for
        # this split. Each subject is one multi-view item.
        # `vis_tol` (depth units): a landmark counts as visible if its camera-z is
        # within this margin of, or in front of, the rendered surface at its pixel.
        # `transform` is an optional callable(sample_dict) -> sample_dict applied
        # before returning (augmentation hook). None = clean data, no aug.
        self.root = Path(root)
        self.subject_ids = list(subject_ids)
        self.depth_scale = depth_scale
        self.vis_tol = vis_tol
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subject_ids)

    # --- schema-aware raw view read -------------------------------------------
    def _read_view(self, vd: Path, new_schema: bool) -> dict:
        """Return the per-view intermediates in a schema-independent form."""
        meta = json.loads((vd / "meta.json").read_text())
        K = np.asarray(meta["K"], dtype=np.float64)                              # (3,3)
        rgb = np.asarray(Image.open(vd / "rgb.png").convert("RGB"), dtype=np.float32) / 255.0
        depth = np.load(vd / "depth.npy").astype(np.float32)                     # (H,W)
        if new_schema:
            Rt = np.asarray(meta["Rt"], dtype=np.float64)                        # (3,4)
            R, t = Rt[:, :3], Rt[:, 3]
            lm_cam = np.load(vd / "landmarks_cam.npy").astype(np.float64)        # (68,3)
            uv = np.asarray(meta["landmark_uv"], dtype=np.float64)              # (68,2)
            P_ref = np.asarray(meta["P"], dtype=np.float64)
        else:
            R = np.asarray(meta["R"], dtype=np.float64)                          # (3,3)
            t = np.asarray(meta["t"], dtype=np.float64).reshape(3)
            lm_cam = np.load(vd / "landmarks_3d.npy").astype(np.float64)         # (68,3) cam frame
            uv = np.load(vd / "landmarks_2d.npy").astype(np.float64)             # (68,2)
            P_ref = None                                                         # not stored
        return {"K": K, "R": R, "t": t, "rgb": rgb, "depth": depth,
                "lm_cam": lm_cam, "uv": uv, "P_ref": P_ref}

    def __getitem__(self, i: int) -> dict:
        subj_dir = self.root / self.subject_ids[i]
        view_dirs = sorted((d for d in subj_dir.iterdir() if d.is_dir()),
                           key=lambda d: int(d.name))
        new_schema = (subj_dir / "landmarks_world.npy").exists()

        raws = [self._read_view(vd, new_schema) for vd in view_dirs]

        # World landmarks: loaded once (new) or derived from view 0 (old). The
        # per-view inversion below cross-checks every view against this.
        if new_schema:
            lm_world = np.load(subj_dir / "landmarks_world.npy").astype(np.float64)  # (68,3)
        else:
            r0 = raws[0]
            lm_world = (r0["lm_cam"] - r0["t"]) @ r0["R"]                        # (68,3)

        rgbd, proj, lm2d, vis = [], [], [], []
        for r in raws:
            K, R, t = r["K"], r["R"], r["t"]
            rgb, depth, lm_cam, uv = r["rgb"], r["depth"], r["lm_cam"], r["uv"]

            face = depth > 0
            filled = binary_fill_holes(face)
            holes = filled & ~face
            med = np.median(depth[face])
            depth = np.where(holes, med, depth)

            depth_n = (depth - med) / self.depth_scale

            # rgb is (H,W,3); move channels first and append depth_n as channel 4.
            x = np.concatenate([rgb.transpose(2, 0, 1), depth_n[None]], axis=0).astype(np.float32)

            # P = K @ [R|t]. New schema also stores meta["P"] -> assert they match.
            P = K @ np.hstack([R, t[:, None]])
            if r["P_ref"] is not None:
                assert np.allclose(P, r["P_ref"], atol=1e-6)

            # Cross-check world landmarks against this view: lm_world = (lm_cam - t) @ R,
            # and their projection through P must reproduce the stored pixel landmarks.
            assert np.allclose((lm_cam - t) @ R, lm_world, atol=1e-3)
            assert np.allclose(_project_np(lm_world, P), uv, atol=1.0)

            # Geometric visibility (we do NOT use any renderer occlusion flag: the
            # z-buffer test has silhouette-edge false negatives). A landmark is
            # visible iff its projection is in-image AND it is at/in-front of the
            # rendered surface (z <= depth + vis_tol), OR it lands on an interior
            # depth hole (the eyeless eye region, where the point is really present).
            H, W = depth.shape
            in_img = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            ui = np.clip(np.round(uv[:, 0]).astype(int), 0, W - 1)
            vi = np.clip(np.round(uv[:, 1]).astype(int), 0, H - 1)
            at_or_front = lm_cam[:, 2] <= depth[vi, ui] + self.vis_tol
            on_hole = holes[vi, ui]
            v_in = (in_img & (on_hole | at_or_front)).astype(np.float32)         # (68,)

            rgbd.append(x)
            proj.append(P)
            lm2d.append(uv)
            vis.append(v_in)

        sample = {
            "rgbd": torch.from_numpy(np.stack(rgbd)).float(),          # (N,4,H,W)
            "proj": torch.from_numpy(np.stack(proj)).float(),          # (N,3,4)
            "landmarks_3d": torch.from_numpy(lm_world).float(),        # (68,3)
            "landmarks_2d": torch.from_numpy(np.stack(lm2d)).float(),  # (N,68,2)
            "vis": torch.from_numpy(np.stack(vis)).float(),            # (N,68)
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
