"""Self-test the render -> point-cloud -> landmark math on the bundled FaceScape
sample TU model (subject 1, neutral). Runs WITHOUT the licensed dataset.

Run with the data venv:
    PYOPENGL_PLATFORM=egl .venv-data/bin/python scripts/data/_selftest_render.py

Validates automatically:
  * EGL offscreen render of the *textured* TU model produces RGB + depth.
  * Depth back-projects to a world point cloud that round-trips back to the
    source pixels (mean reprojection error < 1 px).
  * The 68 landmark vertices project onto the rendered face.

Emits eyeball artifacts to data/facescape/_selftest/:
  panel.png (rgb | depth_vis | landmark overlay), cloud.ply, landmarks.ply
"""

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
TOOLKIT = REPO / "third_party" / "facescape_toolkit" / "toolkit"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(TOOLKIT))

import src.renderer as renderer  # noqa: E402  (toolkit renderer)

from multi_view.data import facescape as fs  # noqa: E402

SAMPLE_OBJ = TOOLKIT.parent / "samples" / "sample_tu_model" / "1_neutral.obj"
LM_NPZ = TOOLKIT / "predef" / "landmark_indices.npz"
OUT = REPO / "data" / "facescape" / "_selftest"


def depth_to_vis(depth: np.ndarray) -> np.ndarray:
    vis = np.zeros(depth.shape, dtype=np.uint8)
    m = depth > 0
    if m.any():
        d = depth[m]
        norm = (d - d.min()) / (np.ptp(d) + 1e-9)
        vis[m] = (255 * (1.0 - norm)).astype(np.uint8)  # near = bright
    return cv2.applyColorMap(vis, cv2.COLORMAP_TURBO) * m[..., None]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    h = w = 1000
    # Frontal CV camera (same intrinsics style as demo_landmark).
    K = np.array([[2000, 0, w / 2 - 0.5], [0, 2000, h / 2 - 0.5], [0, 0, 1]], dtype=np.float64)
    Rt = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 600]], dtype=np.float64)

    # Textured TU mesh. The FaceScape .mtl ships Kd=0 (black diffuse factor),
    # which would multiply the texture to black, so reset diffuse to white.
    mesh = trimesh.load(str(SAMPLE_OBJ), process=False)
    mesh.visual.material.diffuse = np.array([255, 255, 255, 255], dtype=np.uint8)
    depth, color_bgr = renderer.render_cvcam(mesh, K, Rt, rend_size=(h, w))
    n_px = int((depth > 0).sum())
    print(f"render: rgb {color_bgr.shape}, depth nonzero px {n_px}")
    assert n_px > 0, "empty render"

    # Back-project depth -> world cloud, then round-trip reproject.
    cloud, pixels = fs.backproject_depth(depth, K, Rt)
    uv = fs.project_points(K, Rt, cloud)
    err = np.linalg.norm(uv - pixels, axis=1)
    print(f"cloud: {len(cloud)} pts | reprojection err mean {err.mean():.4f}px max {err.max():.4f}px")
    assert err.mean() < 1.0, "back-projection round-trip failed"

    # Landmarks: ordered verts -> project.
    verts = fs.load_obj_vertices(SAMPLE_OBJ)
    lm_idx = fs.load_landmark_indices(LM_NPZ, "v10")
    print(f"verts {verts.shape}, landmarks {len(lm_idx)} (max idx {lm_idx.max()})")
    lm_world = verts[lm_idx]
    lm_uv = fs.project_points(K, Rt, lm_world)
    inside = ((lm_uv[:, 0] >= 0) & (lm_uv[:, 0] < w) & (lm_uv[:, 1] >= 0) & (lm_uv[:, 1] < h)).sum()
    print(f"landmarks inside frame: {inside}/{len(lm_idx)}")
    assert inside >= len(lm_idx) - 2, "landmarks fall outside the rendered frame"

    # ---- eyeball artifacts ----
    rgb = color_bgr[:, :, ::-1]
    cv2.imwrite(str(OUT / "rgb.png"), color_bgr)
    dvis = depth_to_vis(depth)
    overlay = color_bgr.copy()
    for (u, v) in lm_uv.round().astype(int):
        cv2.circle(overlay, (u, v), 5, (0, 0, 255), -1)
    panel = np.concatenate([color_bgr, dvis, overlay], axis=1)
    cv2.imwrite(str(OUT / "panel.png"), panel)

    colors = rgb[pixels[:, 1], pixels[:, 0]]
    trimesh.PointCloud(cloud, colors=colors).export(str(OUT / "cloud.ply"))
    lm_colors = np.tile([[255, 0, 0]], (len(lm_world), 1))
    trimesh.PointCloud(lm_world, colors=lm_colors).export(str(OUT / "landmarks.ply"))

    print("\nAUTOMATIC CHECKS PASSED")
    print(f"eyeball: {OUT/'panel.png'}  (rgb | depth | landmarks)")
    print(f"3D     : {OUT/'cloud.ply'} + {OUT/'landmarks.ply'}")


if __name__ == "__main__":
    main()
