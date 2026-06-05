"""FaceScape multi-view reader and geometry helpers.

This module encodes the facts established from the FaceScape toolkit
(github.com/zhuhao-nju/facescape) so the rest of the pipeline does not have to
re-derive conventions:

Coordinate frames
-----------------
* Multi-view ``params.json`` uses the **CV camera** convention. ``Rt`` is the
  world->camera extrinsic ``[R|t]`` (3x4): ``x_cam = R @ x_world + t``.
* The raw multi-view ``.ply`` scan lives in the **world** frame -- the cameras
  project it directly (this is what ``demo_mview_projection`` renders).
* The **TU model** (``models_reg/<id>_<exp>.obj``, 26317 verts) lives in a
  per-subject **canonical** frame. ``Rt_scale_dict[id][exp] = [scale, Rt_cw]``
  maps **world -> canonical**::

      x_can = scale * (R_cw @ x_world) + t_cw          (demo_align)

  We render the *textured* TU model through the real cameras and read landmarks
  off it, so we need the inverse, **canonical -> world**::

      x_world = (1 / scale) * R_cw.T @ (x_can - t_cw)

  The toolkit notes scan<->TU alignment has "minor misalignment"; landmark GT is
  therefore accurate to that tolerance.

The 68 landmark vertex indices live in ``predef/landmark_indices.npz`` under key
``'v10'`` for the TU model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

# Number of per-view keys in params.json (K, Rt, distortion, width, height,
# matches, valid, sn, ori) -> used to recover the view count.
_KEYS_PER_VIEW = 9


# --------------------------------------------------------------------------- #
# Camera parameters (params.json)
# --------------------------------------------------------------------------- #
@dataclass
class ViewParams:
    """Calibrated parameters for one camera view."""

    vid: int
    K: np.ndarray  # (3, 3) intrinsics
    Rt: np.ndarray  # (3, 4) world->camera extrinsics (CV convention)
    dist: np.ndarray  # (5,) distortion k1 k2 p1 p2 k3
    width: int
    height: int
    valid: bool
    sn: str = ""

    @property
    def R(self) -> np.ndarray:
        return self.Rt[:, :3]

    @property
    def t(self) -> np.ndarray:
        return self.Rt[:, 3]

    @property
    def center(self) -> np.ndarray:
        """Camera center in world coordinates: C = -R^T t."""
        return -self.R.T @ self.t


def load_params(path: str | Path) -> dict:
    """Load a multi-view ``params.json`` file."""
    with open(path, "r") as f:
        return json.load(f)


def num_views(params: dict) -> int:
    return len(params) // _KEYS_PER_VIEW


def get_view(params: dict, vid: int) -> ViewParams:
    """Extract one view's parameters. ``K`` is reduced to the top-left 3x3."""
    K = np.asarray(params["%d_K" % vid], dtype=np.float64)[:3, :3]
    Rt = np.asarray(params["%d_Rt" % vid], dtype=np.float64)
    dist = np.asarray(params["%d_distortion" % vid], dtype=np.float64)
    return ViewParams(
        vid=vid,
        K=K,
        Rt=Rt,
        dist=dist,
        width=int(params["%d_width" % vid]),
        height=int(params["%d_height" % vid]),
        valid=bool(params["%d_valid" % vid]),
        sn=str(params.get("%d_sn" % vid, "")),
    )


def view_ids(params: dict, valid_only: bool = True) -> list[int]:
    """Return view indices, optionally filtered to ``*_valid == True``."""
    ids = range(num_views(params))
    if not valid_only:
        return list(ids)
    return [v for v in ids if bool(params["%d_valid" % v])]


def sample_views(
    params: dict,
    n: int = 5,
    ids: Sequence[int] | None = None,
    valid_only: bool = True,
) -> list[int]:
    """Pick ``n`` views spread evenly in azimuth around the head.

    If ``ids`` is given it is returned as-is (explicit selection). Otherwise the
    valid views are sorted by the azimuth of their camera center (atan2 in the
    world XZ plane about the centroid of all camera centers) and ``n`` are taken
    at even rank intervals, biased to include the most frontal-ish spread.
    """
    if ids is not None:
        return list(ids)

    vids = view_ids(params, valid_only=valid_only)
    if n >= len(vids):
        return vids

    centers = np.stack([get_view(params, v).center for v in vids])
    centroid = centers.mean(axis=0)
    rel = centers - centroid
    azimuth = np.arctan2(rel[:, 0], rel[:, 2])  # angle in world XZ plane
    order = np.argsort(azimuth)
    picks = np.linspace(0, len(vids) - 1, n).round().astype(int)
    return [vids[order[p]] for p in picks]


# --------------------------------------------------------------------------- #
# Projection / back-projection (CV camera)
# --------------------------------------------------------------------------- #
def project_points(K: np.ndarray, Rt: np.ndarray, pts_world: np.ndarray) -> np.ndarray:
    """Project Nx3 world points to Nx2 pixel coordinates (CV pinhole, no distortion)."""
    pts = np.asarray(pts_world, dtype=np.float64)
    cam = pts @ Rt[:, :3].T + Rt[:, 3]  # world->camera
    uv = cam @ K.T
    uv = uv[:, :2] / uv[:, 2:3]
    return uv


def backproject_depth(
    depth: np.ndarray, K: np.ndarray, Rt: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a rendered depth map to a world-frame point cloud.

    ``depth`` is the linear camera-space z returned by ``renderer.render_cvcam``
    (0 where there is no geometry). Returns ``(points_world (M,3),
    pixels (M,2) as integer (u, v))`` for the M valid pixels.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    ys, xs = np.nonzero(depth > 0)
    d = depth[ys, xs]
    x_cam = (xs - cx) / fx * d
    y_cam = (ys - cy) / fy * d
    cam = np.stack([x_cam, y_cam, d], axis=1)  # CV camera frame
    R, t = Rt[:, :3], Rt[:, 3]
    world = (cam - t) @ R  # R^T @ (cam - t), vectorized
    pixels = np.stack([xs, ys], axis=1)
    return world, pixels


# --------------------------------------------------------------------------- #
# TU model + canonical<->world alignment
# --------------------------------------------------------------------------- #
def load_obj_vertices(path: str | Path) -> np.ndarray:
    """Parse ``v x y z`` lines of an .obj in file order (Nx3).

    We parse directly rather than via trimesh because trimesh's default
    processing reorders/merges vertices, which would break ``landmark_indices``.
    """
    verts: list[list[float]] = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(verts, dtype=np.float64)


def load_rt_scale(path: str | Path, subject_id: int, exp_id: int) -> tuple[float, np.ndarray]:
    """Return ``(scale, Rt_cw (3x4))`` mapping world -> canonical for an id/exp."""
    with open(path, "r") as f:
        d = json.load(f)
    entry = d["%d" % subject_id]["%d" % exp_id]
    return float(entry[0]), np.asarray(entry[1], dtype=np.float64)


def canonical_to_world(verts_can: np.ndarray, scale: float, Rt_cw: np.ndarray) -> np.ndarray:
    """Map canonical-frame points to world frame: x_world = (1/s) R_cw^T (x_can - t_cw)."""
    R, t = Rt_cw[:, :3], Rt_cw[:, 3]
    return ((np.asarray(verts_can, dtype=np.float64) - t) @ R) / scale


def load_landmark_indices(npz_path: str | Path, key: str = "v10") -> np.ndarray:
    """Load the 68 TU-model landmark vertex indices (key 'v10' for the TU model)."""
    return np.load(npz_path)[key]


# --------------------------------------------------------------------------- #
# 6-DoF head pose
# --------------------------------------------------------------------------- #
def head_pose_canonical_to_camera(
    scale: float, Rt_cw: np.ndarray, Rt_cam: np.ndarray
) -> dict:
    """Compose the canonical-head -> camera similarity for one view.

    Returns a dict with a pure rotation ``R`` (3x3), translation ``t`` (3),
    isotropic ``scale`` ``s`` and unit quaternion ``quat`` (w, x, y, z), such that
    ``x_cam = s * R @ x_can + t``. The 6-DoF head pose is ``(R, t)``; ``s`` carries
    the canonical<->world scale and is reported separately.
    """
    R_cw, t_cw = Rt_cw[:, :3], Rt_cw[:, 3]
    R_cam, t_cam = Rt_cam[:, :3], Rt_cam[:, 3]
    s = 1.0 / scale
    R = R_cam @ R_cw.T
    t = t_cam - s * (R @ t_cw)
    return {
        "R": R.tolist(),
        "t": t.tolist(),
        "scale": s,
        "quat": _rotmat_to_quat(R).tolist(),
    }


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z), no scipy dependency."""
    m = np.asarray(R, dtype=np.float64)
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)
