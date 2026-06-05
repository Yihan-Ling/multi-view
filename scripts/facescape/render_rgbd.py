"""Generate virtual multi-view RGB-D + world-frame point clouds (with facial
landmarks attached) from FaceScape TU models.

For each subject we render the *textured* TU model (placed in the multi-view world
frame via the inverse Rt_scale transform) through N sampled cameras, and per view
emit the model-ready RGB-D plus the final 3D bundle:

    rgbd/<id>/<view>/
        rgb.png  depth.npy  mask.png            # RGB-D (train)
        cloud_world.npy  landmarks_world.npy     # 3D output (world frame)
        meta.json                                # K, Rt, P, bbox, 6-DoF pose, ...
        depth_vis.png panel.png cloud.ply landmarks.ply lmk_overlay.png  # display
    rgbd/<id>/tuple_index.json                    # groups the views as one sample

Two modes:
  dataset : real subjects -- needs TU obj + params.json + Rt_scale (licensed data).
  demo    : the bundled sample TU model + generated ring cameras (no license),
            used to smoke-test the full output bundle end-to-end.

Run (data venv, EGL):
    PYOPENGL_PLATFORM=egl .venv-data/bin/python scripts/data/render_rgbd.py --mode demo
"""

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
TOOLKIT = REPO / "third_party" / "facescape_toolkit" / "toolkit"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(TOOLKIT))

import src.renderer as renderer  # noqa: E402

from multi_view.data import facescape as fs  # noqa: E402

PREDEF = TOOLKIT / "predef"
LM_NPZ = PREDEF / "landmark_indices.npz"
RT_SCALE = PREDEF / "Rt_scale_dict.json"
SAMPLE_TU = TOOLKIT.parent / "samples" / "sample_tu_model" / "1_neutral.obj"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def depth_to_vis(depth: np.ndarray) -> np.ndarray:
    vis = np.zeros(depth.shape, dtype=np.uint8)
    m = depth > 0
    if m.any():
        d = depth[m]
        vis[m] = (255 * (1.0 - (d - d.min()) / (np.ptp(d) + 1e-9))).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_TURBO) * m[..., None]


def look_at_cv(eye: np.ndarray, target: np.ndarray, up=(0, 1, 0)) -> np.ndarray:
    """Build a CV-camera extrinsic [R|t] (world->camera) looking from eye at target."""
    eye, target, up = map(lambda v: np.asarray(v, float), (eye, target, up))
    zc = target - eye
    zc /= np.linalg.norm(zc)
    xc = np.cross(zc, up)
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R = np.stack([xc, yc, zc])  # rows
    t = -R @ eye
    return np.concatenate([R, t[:, None]], axis=1)


def load_tu_world(obj_path, scale, Rt_cw):
    """Load the textured TU mesh and ordered verts, both mapped canonical->world.

    ``scale``/``Rt_cw`` may be None to render in the mesh's own frame (demo).
    Returns ``(trimesh_world, ordered_verts_world)``.
    """
    mesh = trimesh.load(str(obj_path), process=False)
    # FaceScape .mtl ships Kd=0 (black); reset diffuse so the texture shows.
    mesh.visual.material.diffuse = np.array([255, 255, 255, 255], dtype=np.uint8)
    verts = fs.load_obj_vertices(obj_path)
    if scale is not None:
        mesh.vertices = fs.canonical_to_world(mesh.vertices, scale, Rt_cw)
        verts = fs.canonical_to_world(verts, scale, Rt_cw)
    return mesh, verts


def write_ply(path, xyz, rgb):
    trimesh.PointCloud(np.asarray(xyz), colors=np.asarray(rgb, np.uint8)).export(str(path))


# --------------------------------------------------------------------------- #
# frame transforms + landmark blobs
# --------------------------------------------------------------------------- #
# CV camera -> display: flip Y/Z so MeshLab shows the face upright and facing the
# viewer from the virtual camera's direction (CV is x-right, y-down, z-forward).
_DISPLAY_FLIP = np.array([1.0, -1.0, -1.0])


def to_camera(pts_world: np.ndarray, Rt: np.ndarray) -> np.ndarray:
    """World -> CV camera frame: x_cam = R @ x_world + t."""
    return np.asarray(pts_world, np.float64) @ Rt[:, :3].T + Rt[:, 3]


def cam_to_display(pts_cam: np.ndarray) -> np.ndarray:
    """CV camera frame -> MeshLab display frame (upright, facing viewer)."""
    return np.asarray(pts_cam, np.float64) * _DISPLAY_FLIP


def red(n: int) -> np.ndarray:
    return np.tile([[255, 0, 0]], (n, 1)).astype(np.uint8)


def render_cloud_png(pts, rgb, out_path, width=1600, margin=20, splat=1, bg=255):
    """Orthographic front-view splat of a display-frame point cloud to a PNG.

    Looks down display -Z (the points are already camera-facing), z-buffered by
    painting far points first so nearer ones overwrite. Pure numpy/cv2 -- no GL.
    """
    pts = np.asarray(pts, np.float64)
    rgb = np.asarray(rgb, np.uint8)
    xy, z = pts[:, :2], pts[:, 2]
    mn, mx = xy.min(0), xy.max(0)
    scale = (width - 2 * margin) / max(mx[0] - mn[0], 1e-9)
    h = int((mx[1] - mn[1]) * scale + 2 * margin)
    u = ((xy[:, 0] - mn[0]) * scale + margin).astype(int)
    v = (h - 1 - ((xy[:, 1] - mn[1]) * scale + margin)).astype(int)  # flip y for image
    order = np.argsort(z)  # far (small z) first -> near overwrites
    u, v, col = u[order], v[order], rgb[order]
    img = np.full((h, width, 3), bg, np.uint8)
    for dy in range(-splat, splat + 1):
        for dx in range(-splat, splat + 1):
            img[np.clip(v + dy, 0, h - 1), np.clip(u + dx, 0, width - 1)] = col
    cv2.imwrite(str(out_path), img[:, :, ::-1])  # RGB -> BGR


# --------------------------------------------------------------------------- #
# per-view render
# --------------------------------------------------------------------------- #
def render_view(mesh_world, K, Rt, rend_size, lm_world, out_dir, vid, *, cloud_stride, src):
    h, w = rend_size
    depth, color_bgr = renderer.render_cvcam(mesh_world, K, Rt, rend_size=(h, w))
    if (depth > 0).sum() == 0:
        raise RuntimeError(f"view {vid}: empty render")
    rgb = color_bgr[:, :, ::-1]
    mask = (depth > 0).astype(np.uint8) * 255

    # point cloud from the depth (subsampled by stride); backproject_depth returns
    # world coords -- we convert to the per-view camera frame below.
    cloud_world, pixels = fs.backproject_depth(depth, K, Rt)
    if cloud_stride > 1:
        cloud_world, pixels = cloud_world[::cloud_stride], pixels[::cloud_stride]
    cloud_rgb = rgb[pixels[:, 1], pixels[:, 0]]

    # landmarks: project + occlusion-aware visibility
    lm_uv = fs.project_points(K, Rt, lm_world)
    lm_camz = (lm_world @ Rt[:, :3].T + Rt[:, 3])[:, 2]
    vis = np.zeros(len(lm_world), bool)
    for i, (u, v) in enumerate(lm_uv.round().astype(int)):
        if 0 <= u < w and 0 <= v < h and depth[v, u] > 0:
            vis[i] = abs(lm_camz[i] - depth[v, u]) < 0.02 * lm_camz[i]

    # face bbox from landmark extent (clipped, with margin)
    inb = (lm_uv[:, 0] >= 0) & (lm_uv[:, 0] < w) & (lm_uv[:, 1] >= 0) & (lm_uv[:, 1] < h)
    pts = lm_uv[inb]
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    mx, my = 0.25 * (x1 - x0), 0.25 * (y1 - y0)
    bbox = [
        float(max(0, x0 - mx)), float(max(0, y0 - my)),
        float(min(w, x1 + mx)), float(min(h, y1 + my)),
    ]

    P = (K @ Rt).tolist()
    pose = fs.head_pose_canonical_to_camera(SCALE_CTX[0], SCALE_CTX[1], Rt) if SCALE_CTX[0] else None

    # per-view 3D in the camera (CV) frame -- sensor-faithful; world is recoverable
    # from Rt and the shared landmarks_world.npy at the subject level.
    cloud_cam = to_camera(cloud_world, Rt)
    lm_cam = to_camera(lm_world, Rt)

    vdir = out_dir / str(vid)
    vdir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(vdir / "rgb.png"), color_bgr)
    np.save(vdir / "depth.npy", depth.astype(np.float32))
    cv2.imwrite(str(vdir / "mask.png"), mask)
    np.save(vdir / "cloud_cam.npy", np.concatenate([cloud_cam, cloud_rgb], axis=1).astype(np.float32))
    np.save(vdir / "landmarks_cam.npy", lm_cam.astype(np.float32))

    meta = {
        "view": int(vid),
        "K": K.tolist(),
        "Rt": Rt.tolist(),
        "P": P,
        "width": int(w),
        "height": int(h),
        "bbox_xyxy": bbox,
        "depth_units": "FaceScape world units (TU/multi-view metric; not calibrated to meters)",
        "frames": {
            "cloud_cam.npy": "camera CV (x-right, y-down, z-forward)",
            "landmarks_cam.npy": "camera CV",
            "landmarks_world.npy": "world, shared across views (at subject dir)",
            "display_ply": "camera, Y/Z-flipped for upright viewer-facing display",
        },
        "landmark_visible": vis.astype(int).tolist(),
        "landmark_uv": lm_uv.astype(float).tolist(),
        "head_pose_can2cam": pose,
        "source_mesh": src,
    }
    with open(vdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # 2D display artifacts
    cv2.imwrite(str(vdir / "depth_vis.png"), depth_to_vis(depth))
    overlay = color_bgr.copy()
    for i, (u, v) in enumerate(lm_uv.round().astype(int)):
        col = (0, 0, 255) if vis[i] else (128, 128, 128)
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(overlay, (u, v), max(2, w // 250), col, -1)
    cv2.imwrite(str(vdir / "lmk_overlay.png"), overlay)
    cv2.imwrite(str(vdir / "panel.png"), np.concatenate([color_bgr, depth_to_vis(depth), overlay], axis=1))

    # 3D display (.ply) in the camera-facing display frame; one red dot per landmark
    face_disp = cam_to_display(cloud_cam)
    lm_disp = cam_to_display(lm_cam)
    lm_rgb = red(len(lm_disp))
    fl_pts = np.concatenate([face_disp, lm_disp], axis=0)
    fl_rgb = np.concatenate([cloud_rgb, lm_rgb], axis=0)
    write_ply(vdir / "face.ply", face_disp, cloud_rgb)
    write_ply(vdir / "landmark.ply", lm_disp, lm_rgb)
    write_ply(vdir / "face-landmark.ply", fl_pts, fl_rgb)
    return vid, int(vis.sum()), fl_pts, fl_rgb


# transform context for 6-DoF pose (set per subject); (scale, Rt_cw) or (None, None)
SCALE_CTX = (None, None)


def render_subject(sid, mesh_world, verts_world, cameras, out_root, *, exp, cloud_stride, src):
    out_dir = out_root / str(sid)
    out_dir.mkdir(parents=True, exist_ok=True)
    lm_idx = fs.load_landmark_indices(LM_NPZ, "v10")
    lm_world = verts_world[lm_idx]

    done = []
    panel_tiles = []  # (display points, colors) per view, for panel.ply
    for vid, K, Rt, rend_size in cameras:
        v, nvis, fl_pts, fl_rgb = render_view(
            mesh_world, K, Rt, rend_size, lm_world, out_dir, vid,
            cloud_stride=cloud_stride, src=src,
        )
        done.append(v)
        panel_tiles.append((fl_pts, fl_rgb))
        print(f"  subject {sid} view {v}: {nvis}/{len(lm_world)} landmarks visible")

    # panel.ply: each view's camera-facing face-landmark cloud, recentered and laid
    # out side by side along display-X so all 5 viewpoints sit in one file.
    spacing = 1.3 * max(float(np.ptp(p[:, 0])) for p, _ in panel_tiles)
    panel_pts, panel_rgb = [], []
    for i, (pts, rgb) in enumerate(panel_tiles):
        shifted = pts - pts.mean(0) + np.array([i * spacing, 0.0, 0.0])
        panel_pts.append(shifted)
        panel_rgb.append(rgb)
    all_pts, all_rgb = np.concatenate(panel_pts), np.concatenate(panel_rgb)
    write_ply(out_dir / "panel.ply", all_pts, all_rgb)
    render_cloud_png(all_pts, all_rgb, out_dir / "panel.png")

    np.save(out_dir / "landmarks_world.npy", lm_world.astype(np.float32))
    with open(out_dir / "tuple_index.json", "w") as f:
        json.dump(
            {
                "subject": int(sid),
                "expression": exp,
                "n_views": len(done),
                "views": [int(v) for v in done],
                "per_view_3d_frame": "camera CV (cloud_cam.npy / landmarks_cam.npy)",
                "landmarks_world": "landmarks_world.npy (shared world frame)",
                "panel": "panel.ply (all views side by side, camera-facing)",
            },
            f,
            indent=2,
        )
    print(f"subject {sid}: wrote {len(done)} views -> {out_dir}")


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def cameras_from_params(params, n_views, long_side, view_ids=None):
    vids = fs.sample_views(params, n=n_views, ids=view_ids)
    cams = []
    for vid in vids:
        vp = fs.get_view(params, vid)
        s = long_side / max(vp.width, vp.height)
        K = vp.K.copy()
        K[:2, :] *= s
        rend = (int(round(vp.height * s)), int(round(vp.width * s)))
        cams.append((vid, K, vp.Rt, rend))
    return cams


def ring_cameras(centroid, dist, n_views, res, fov_deg=50.0):
    """Generate n virtual CV cameras on a frontal arc for demo mode."""
    f = 0.5 * res / np.tan(np.radians(fov_deg) / 2)
    K = np.array([[f, 0, res / 2 - 0.5], [0, f, res / 2 - 0.5], [0, 0, 1]])
    cams = []
    for i, az in enumerate(np.linspace(-50, 50, n_views)):
        a = np.radians(az)
        eye = centroid + dist * np.array([np.sin(a), 0.0, np.cos(a)])
        cams.append((i, K, look_at_cv(eye, centroid), (res, res)))
    return cams


def run_demo(args):
    global SCALE_CTX
    SCALE_CTX = (None, None)  # demo renders in the mesh's own frame
    mesh, verts = load_tu_world(SAMPLE_TU, None, None)
    centroid = verts.mean(0)
    dist = 2.1 * np.linalg.norm(verts.max(0) - verts.min(0))
    cams = ring_cameras(centroid, dist, args.n_views, args.res)
    out = REPO / "data" / "facescape" / "rgbd_demo"
    render_subject(1, mesh, verts, cams, out, exp="1_neutral", cloud_stride=args.cloud_stride,
                   src=str(SAMPLE_TU.relative_to(REPO)))


def run_dataset(args):
    global SCALE_CTX
    raw = Path(args.raw_root)
    out = REPO / "data" / "facescape" / "rgbd"
    exp_name = f"{args.exp}_neutral" if args.exp == 1 else str(args.exp)
    subjects = [int(s) for s in args.subjects.split(",")] if args.subjects else _selection_ids()
    for sid in subjects:
        tu_obj = raw / "tu" / str(sid) / f"{exp_name}.obj"
        params_path = raw / "mview" / str(sid) / exp_name / "params.json"
        if not tu_obj.exists() or not params_path.exists():
            print(f"subject {sid}: missing {tu_obj if not tu_obj.exists() else params_path} -- skip")
            continue
        scale, Rt_cw = fs.load_rt_scale(RT_SCALE, sid, args.exp)
        SCALE_CTX = (scale, Rt_cw)
        mesh, verts = load_tu_world(tu_obj, scale, Rt_cw)
        params = fs.load_params(params_path)
        cams = cameras_from_params(params, args.n_views, args.long_side)
        render_subject(sid, mesh, verts, cams, out, exp=exp_name,
                       cloud_stride=args.cloud_stride, src=str(tu_obj.relative_to(REPO)))


def _selection_ids():
    sel = REPO / "data" / "facescape" / "selection.json"
    if sel.exists():
        return [int(s["id"]) for s in json.load(open(sel))["subjects"]]
    raise SystemExit("no --subjects and no data/facescape/selection.json found")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["dataset", "demo"], default="demo")
    ap.add_argument("--subjects", default="", help="dataset: comma-separated ids (else selection.json)")
    ap.add_argument("--raw-root", default=str(REPO / "data" / "facescape" / "raw"))
    ap.add_argument("--exp", type=int, default=1, help="expression id (1 = neutral)")
    ap.add_argument("--n-views", type=int, default=5)
    ap.add_argument("--long-side", type=int, default=1024, help="dataset render long side (px)")
    ap.add_argument("--res", type=int, default=512, help="demo render resolution (px)")
    ap.add_argument("--cloud-stride", type=int, default=4, help="subsample factor for point clouds")
    args = ap.parse_args()
    (run_demo if args.mode == "demo" else run_dataset)(args)


if __name__ == "__main__":
    main()
