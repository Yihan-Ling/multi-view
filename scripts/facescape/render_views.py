#!/usr/bin/env python3
"""Render the FaceScape TU model through virtual cameras -> per-view RGB + depth + landmarks.

Reference facts (data truths, not design choices you need to make):

  Input per subject (produced by extract.py):
      data/facescape/<id_range>/<id>/1_neutral.obj      TU mesh, ~26k verts, ORDER PRESERVED
                                     1_neutral.jpg      4K texture
                                     1_neutral.obj.mtl  material -- ships Kd=0 (renders BLACK)

  Landmarks: third_party/facescape_toolkit/toolkit/predef/landmark_indices.npz, key 'v10'
      -> 68 ints indexing into mesh.vertices. They are ONLY valid if vertex order is kept,
         which is why we load with process=False below.

  Coordinate conventions -- the #1 source of bugs in this file:
    - OpenCV (what we compute GT in):  x_cam = R @ x_world + t,  camera looks +Z, +X right,
      +Y down.  K = [[fx,0,cx],[0,fy,cy],[0,0,1]].  Projection: u = fx*x/z + cx, v = fy*y/z + cy.
    - OpenGL / pyrender:  camera looks -Z, +Y up.  So the 4x4 pose handed to pyrender is the
      camera->world transform with Y and Z flipped:  pose_gl = c2w @ diag(1,-1,-1,1).
    - pyrender's depth is linear z-distance in camera space (0.0 = no geometry hit). Same
      magnitude as the OpenCV z, so it back-projects with the OpenCV intrinsics directly.

  TU canonical frame (read off the mesh, not a choice): +Y is up, +Z is forward (the face
      points +Z; nose tip is the Z-max, back of head is the Z-min), X is lateral. So the
      world "up" for look-at is (0,1,0), and head orientation is a yaw about +Y.

  Units: FaceScape world units (~mm-scale, NOT calibrated meters). The sample mesh spans
      roughly +/-100 in x/y, so virtual cameras live a few hundred units from the head.

  Output (what you are building toward), per (id, camera):
      data/facescape/virtual_camera_data/<id>/<cam_name>/
          rgb.png                         8-bit color
          depth.npy                       float32 (H,W), world units, 0 = miss
          cloud.ply                       colored point cloud (back-projected depth, camera frame)
          rgb_landmark_overlay.png        rgb + 68 dots          (eyeball only)
          landmarks_2d.npy                (68,2) float pixel (u,v)         -- 2D target
          landmarks_3d.npy                (68,3) float, CAMERA frame        -- 3D target
          meta.json                       camera K/R/t + 6-DoF head pose in camera frame
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import cv2
import trimesh
import pyrender          # run with PYOPENGL_PLATFORM=egl for headless GPU
from PIL import Image


@dataclass
class Camera:
    """One virtual camera: name + image size + OpenCV intrinsics/extrinsics
    (world->cam, x_cam = R @ x_world + t). Both producers (load_camera,
    default_ring) emit these, and render/project_landmarks consume them, so the
    five fields that always travel together stay bundled as one object."""
    id: str
    W: int
    H: int
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray

    def __post_init__(self):
        # JSON / list inputs -> arrays, once, no matter who constructs the Camera.
        self.K = np.asarray(self.K, dtype=float)
        self.R = np.asarray(self.R, dtype=float)
        self.t = np.asarray(self.t, dtype=float)


# ---------------------------------------------------------------------------
# Reference geometry: a "look-at" camera in the OpenCV convention.
# Tedious to get the handedness right, so here it is fully. Given the camera
# position `eye`, the `target` it points at, and a world `up` direction, it
# returns (R, t) for world->camera with x_cam = R @ x_world + t (camera +Z
# forward, +Y down, +X right). Use this to GENERATE test cameras around a head.
# ---------------------------------------------------------------------------
def look_at_cv(eye, target, up=(0.0, 1.0, 0.0)):   # +Y up in the TU frame
    eye, target, up = (np.asarray(v, dtype=float) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)            # +Z: forward, toward the target
    x = np.cross(z, up)
    x /= np.linalg.norm(x)            # +X: right  (cross(z,up), so world-up stays up)
    y = np.cross(z, x)                # +Y: down  (right-handed: y = z x x)
    R_c2w = np.stack([x, y, z], axis=1)   # columns are the camera axes in world coords
    R = R_c2w.T                       # world->camera rotation
    t = -R @ eye                      # world->camera translation
    return R, t


def intrinsics_from_fov(fov_deg, W, H):
    """Square-pixel K from a horizontal field of view. Handy for test cameras."""
    fx = fy = (W / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    return np.array([[fx, 0, W / 2.0], [0, fy, H / 2.0], [0, 0, 1.0]])


# ---------------------------------------------------------------------------
# Reference: 3x3 rotation -> unit quaternion [w, x, y, z]. The branch-on-trace
# form below is the standard numerically-stable recipe; the sign placement is a
# classic trap, so it's given fully (like look_at_cv). Use it on the head's
# rotation-in-camera-frame to store pose compactly.
# ---------------------------------------------------------------------------
def rotmat_to_quat(R):
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0                 # s = 4*w
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0   # s = 4*x
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0   # s = 4*y
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0   # s = 4*z
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


class ViewRenderer:
    LANDMARK_NPZ = Path(
        "third_party/facescape_toolkit/toolkit/predef/landmark_indices.npz"
    )

    def __init__(self,
                 data_root="data/facescape",
                 out_root="data/facescape/virtual_camera_data"):
        self.data_root = Path(data_root)
        self.out_root = Path(out_root)
        self.lm_idx = np.load(self.LANDMARK_NPZ)["v10"]   # (68,) vertex indices

    # ---- Stage A: load the TU model -------------------------------------
    # WORKED for you because both lines here are traps (see module docstring).
    
    def find_id_range_folder(self, id):
        n = int(id)
        for d in self.data_root.iterdir():
            parts = d.name.split("_")          # a bucket is exactly <digits>_<digits>
            if not d.is_dir() or len(parts) != 2 or not all(p.isdigit() for p in parts):
                continue                       # skips _selftest, rgbd_demo, virtual_camera_data, ...
            lo, hi = int(parts[0]), int(parts[1])
            if lo <= n <= hi:
                return d.name
            
        raise FileNotFoundError(f"no folder contains id {id} under {self.data_root}")
    
    def load_model(self, id, exp="1_neutral"):
        id_range_folder = self.find_id_range_folder(id)
        obj = self.data_root / id_range_folder / id / f"{exp}.obj"
        mesh = trimesh.load(obj, process=False)   # process=False keeps vertex order
        if isinstance(mesh, trimesh.Scene):       # rare, but guard it
            mesh = mesh.dump(concatenate=True)

        # Kd=0 in the .mtl -> texture * 0 -> black. Reset the base color to white.
        # (Run `print(type(mesh.visual.material))` once to see which attr your
        #  trimesh version exposes -- baseColorFactor for PBR, diffuse for Simple.)
        mat = mesh.visual.material
        if hasattr(mat, "baseColorFactor"):
            mat.baseColorFactor = np.array([255, 255, 255, 255], dtype=np.uint8)
        if hasattr(mat, "diffuse"):
            mat.diffuse = np.array([255, 255, 255, 255], dtype=np.uint8)

        # Landmark trap: this .obj has 26317 'v' lines but 26404 'vt' lines, so
        # trimesh splits seam vertices to give each one a unique UV -> mesh.vertices
        # is reordered/expanded and the landmark indices no longer line up. The
        # indices reference the RAW 'v' order, so parse those lines by hand and carry
        # the 68 landmark world-points on the mesh. orient_head rotates them too, so
        # project_landmarks just reads mesh.metadata["lm_world"].
        raw = np.array([ln.split()[1:4] for ln in obj.read_text().splitlines()
                        if ln.startswith("v ")], dtype=float)   # (26317, 3), file order
        mesh.metadata["lm_world"] = raw[self.lm_idx]            # (68, 3)
        return mesh

    # ---- Stage A.5: orient the head (roll / pitch / yaw) ----------------
    # WORKED because rotating about the head's pivot (not the world origin) is a
    # trap. The 68 landmark world-points ride along in mesh.metadata["lm_world"]
    # (set in load_model); we apply the SAME transform to them here, so orientation
    # flows into your projected 2D landmarks and the 6-DoF GT for free.
    #
    # Standard Tait-Bryan roll/pitch/yaw, right-hand rule on the head's anatomical
    # axes (the conventional definitions, from the subject's own frame):
    #     +yaw   -> turns to the subject's RIGHT
    #     +pitch -> face tilts UP   (chin up / gaze up)
    #     +roll  -> tilts to the subject's RIGHT (right ear toward right shoulder)
    # 0,0,0 = facing the front camera (a camera out on +Z looking back at the head).
    #
    # The TU frame is +Y up, +Z forward (gaze), +X the subject's LEFT, so the
    # convention maps onto these elemental rotations with the sign flips below.
    def orient_head(self, mesh, roll=0.0, pitch=0.0, yaw=0.0):
        pivot = mesh.vertices.mean(axis=0)     # head origin = centroid, in world coords
        if roll == 0 and pitch == 0 and yaw == 0:
            mesh.metadata["head_R"] = np.eye(3)   # head->world rotation (identity at 0,0,0)
            mesh.metadata["head_t"] = pivot       # head origin in world frame
            return mesh
        a_pitch = np.radians(-pitch)   # +pitch = up               (nose toward +Y)
        a_yaw = np.radians(-yaw)       # +yaw   = subject's right   (nose toward -X)
        a_roll = np.radians(roll)      # +roll  = subject's right   (top toward -X)
        cx, sx = np.cos(a_pitch), np.sin(a_pitch)
        cy, sy = np.cos(a_yaw), np.sin(a_yaw)
        cz, sz = np.cos(a_roll), np.sin(a_roll)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])   # pitch about X (lateral)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])   # yaw   about Y (up)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])   # roll  about Z (forward)
        R = Ry @ Rx @ Rz                       # yaw-pitch-roll intrinsic order
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pivot - R @ pivot           # rotate about `pivot`, not the origin
        mesh.apply_transform(T)                # mutates in place -> reload per orientation
        lm = mesh.metadata["lm_world"]         # carry the landmarks through the same transform
        mesh.metadata["lm_world"] = (R @ lm.T).T + T[:3, 3]
        # Head pose GT: head->world is rotation R about the centroid, and the centroid
        # is fixed under that rotation, so head_t stays at `pivot`. run() composes these
        # with each camera's (R_c, t_c) to get the 6-DoF head pose in the camera frame.
        mesh.metadata["head_R"] = R            # head->world rotation
        mesh.metadata["head_t"] = pivot        # head origin (centroid) in world
        return mesh

    # ---- helper: OpenCV (R,t) -> pyrender's OpenGL pose -----------------
    # WORKED because this conversion is the convention trap. c2w is camera->world;
    # the diag flip turns the OpenCV camera (+Z fwd, +Y down) into OpenGL's.
    def _gl_pose(self, R, t):
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t
        return c2w @ np.diag([1.0, -1.0, -1.0, 1.0])

    # ---- Stage B: render one view --------------------------------------
    # WORKED: the pyrender scene assembly is boilerplate you don't need to learn.
    def render(self, mesh, cam):
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.4, 0.4, 0.4])
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

        # znear/zfar matter: FaceScape units are ~mm-scale, so the head sits a few
        # hundred units away. pyrender defaults to zfar=100, which clips the whole
        # head away (empty render). Push the far plane well past the head.
        pcam = pyrender.IntrinsicsCamera(fx=cam.K[0, 0], fy=cam.K[1, 1],
                                         cx=cam.K[0, 2], cy=cam.K[1, 2],
                                         znear=1.0, zfar=5000.0)
        pose = self._gl_pose(cam.R, cam.t)
        scene.add(pcam, pose=pose)
        # light from the camera direction so the face is lit, not silhouetted
        scene.add(pyrender.DirectionalLight(intensity=3.0), pose=pose)

        renderer = pyrender.OffscreenRenderer(cam.W, cam.H)
        color, depth = renderer.render(scene)     # (H,W,3) uint8, (H,W) float32
        renderer.delete()
        return color, depth

    # ---- Stage C: project the 68 landmarks to 2D pixels -----------------
    # YOUR TURN. You have the math in the docstring. Steps:
    #   1. grab the 68 world points:           pts = mesh.vertices[self.lm_idx]   (68,3)
    #   2. transform to camera frame:          x_cam = (R @ pts.T).T + t          (68,3)
    #   3. perspective divide with K:          u = fx*x/z + cx,  v = fy*y/z + cy
    #   4. return an (68,2) array of (u,v) pixel coords (floats are fine).
    # (Optional later: an occlusion test comparing z_cam against depth[v,u].)
    def project_landmarks(self, mesh, cam):
        pts = mesh.metadata["lm_world"]   # (68,3) carried from the raw 'v' order
        x_cam = (cam.R@pts.T).T + cam.t
        fx, fy = cam.K[0,0], cam.K[1,1]
        cx, cy = cam.K[0,2], cam.K[1,2]
        x, y, z = x_cam.T
        u = fx*x/z + cx
        v = fy*y/z + cy
        return np.stack([u, v], axis=1)
        

    # ---- Stage C.5: back-project depth -> colored point cloud -----------
    # YOUR TURN. This is the INVERSE of project_landmarks: there you went 3D->2D
    # with K; here you undo it, lifting each hit pixel back to a 3D point in the
    # CAMERA frame from its depth. The per-pixel plumbing (grid + hit mask) is
    # given; you write the three pinhole-inverse lines.
    # (To put the cloud in the WORLD frame instead -- so clouds from different
    #  cameras OVERLAP into one head, a great multi-view sanity check -- apply the
    #  inverse extrinsics afterward:  p_world = R.T @ (p_cam - t).)
    def backproject(self, depth, K, color=None):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        H, W = depth.shape
        vv, uu = np.mgrid[0:H, 0:W]          # vv = row index (v), uu = col index (u)
        mask = depth > 0                     # hit pixels only (0 = miss)
        u = uu[mask].astype(float)           # (N,)
        v = vv[mask].astype(float)           # (N,)
        z = depth[mask]                      # (N,) camera-space distance
        x = (u-cx)/fx*z
        y = (v-cy)/fy*z
        pts = np.stack([x,y,z], axis = 1)
        colors = color[mask] if color is not None else None   # (N,3) RGB, aligned to pts
        return pts, colors

    # ---- Stage D: visualizations + write --------------------------------
    # YOUR TURN. Suggested helpers to write:
    #   - draw_landmarks(img, uv): cv2.circle each (u,v) onto a copy of img.
    #   - write_view(out_dir, color, depth, uv): mkdir out_dir, then save the three
    #     files listed in the module docstring (np.save for depth.npy, PIL/cv2 for pngs).
    
    def draw_landmarks(self, img, landmarks):
        new_img = img.copy()
        for landmark in landmarks:
            cv2.circle(img=new_img, center=(int(landmark[0]), int(landmark[1])), radius=2, color=(255,0,0), thickness=-1)
        return new_img
            
    def write_view(self, out_dir, color, depth, landmarks):
        # color: (H,W,3) uint8 in RGB order (pyrender's output). depth: (H,W) float32.
        # uv: (68,2) pixel coords from project_landmarks.
        #
        # The one trap: channel order on save. pyrender gives RGB; PIL writes RGB.
        # cv2.imwrite expects BGR, so saving this array with cv2 swaps red<->blue
        # (face goes bluish, your red dots go blue). Keep ALL pngs on PIL here so
        # nothing swaps -- your draw_landmarks' (255,0,0) then shows as real red.
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        Image.fromarray(color).save(out_dir / "rgb.png")
        np.save(out_dir / "depth.npy", depth)
        landmark_overlay = self.draw_landmarks(color, landmarks)
        Image.fromarray(landmark_overlay).save(out_dir/"rgb_landmark_overlay.png") 

    # ---- driver ---------------------------------------------------------
    # YOUR TURN. Tie it together:
    #   - parse the id range (e.g. "801-805") into a list of id strings.
    #   - load the camera list (see load_cameras below).
    #   - for each id: load_model, orient_head(mesh, roll, pitch, yaw), then for
    #     each camera render/project/write into out_root/<id>/<cam_name>/.
    #   (orient_head mutates the mesh, so reload per orientation if you ever loop
    #    over several.)
    #
    # GT per (id, camera) -- the payoff. orient_head stashed head->world on the mesh:
    #     mesh.metadata["head_R"]  (3x3)   mesh.metadata["head_t"]  (3,)
    # For one camera with OpenCV extrinsics R_c (3x3), t_c (3,) world->cam:
    #     uv     = self.project_landmarks(mesh, K, R_c, t_c)              # (68,2) 2D target
    #     lm_w   = mesh.metadata["lm_world"]                             # (68,3) world
    #     lm_cam = (R_c @ lm_w.T).T + t_c                                # (68,3) 3D target
    #   Head pose in THIS camera's frame (compose world->cam with head->world):
    #     R_ch = R_c @ mesh.metadata["head_R"]
    #     t_ch = R_c @ mesh.metadata["head_t"] + t_c     # = head centroid in cam coords
    #     quat = rotmat_to_quat(R_ch)                    # [w,x,y,z]
    #   Colored point cloud (the RGB-D as 3D), camera frame:
    #     pts, cols = self.backproject(depth, K, color=color)
    #     trimesh.PointCloud(vertices=pts, colors=cols).export(out_dir / "cloud.ply")
    #   Then write_view(...) for the images + np.save uv and lm_cam, and a meta.json:
    #     {name, W, H, K, R, t, head_quat, head_t_cam, orientation_deg, units:"facescape_world"}
    #   JSON trap: it can't serialize numpy -> call .tolist() on every array first.
    def run(self, id_range, cameras, orientation=(0.0, 0.0, 0.0)):
        roll, pitch, yaw = orientation
        
        # parse id_range
        if "-" in id_range:
            l, r = id_range.split("-")
            ids = [str(i) for i in range(int(l), int(r)+1)]
        else:
            ids = [id_range]
        
        for id in ids:
            mesh = self.load_model(id=id)
            self.orient_head(mesh, roll, pitch, yaw)
            lm_w = mesh.metadata["lm_world"]    # landmark in world frame
            cams = load_camera(cameras) if cameras else default_ring(mesh)
            
            for cam in cams:
                color, depth = self.render(mesh, cam)
                landmarks = self.project_landmarks(mesh, cam)
                pts, cols = self.backproject(depth=depth, K=cam.K, color=color)
                lm_cam = (cam.R @ lm_w.T).T + cam.t     # landmark in camera frame
                R_ch   = cam.R @ mesh.metadata["head_R"]    # head orientation in camera frame
                t_ch   = cam.R @ mesh.metadata["head_t"] + cam.t    # head position in camera frame
                quat   = rotmat_to_quat(R_ch)   # R_ch in quaternion
                
                # write data
                out_dir = self.out_root / id / cam.id
                self.write_view(out_dir=out_dir, color=color, depth=depth, landmarks=landmarks)
                trimesh.PointCloud(vertices=pts, colors=cols).export(out_dir / "rgbd.ply")
                lm_dots = np.tile([255, 0, 0], (len(lm_cam), 1))
                all_pts = np.vstack([pts, lm_cam])
                all_colors = np.vstack([cols, lm_dots])
                trimesh.PointCloud(vertices=all_pts, colors=all_colors).export(out_dir / "rgbd_landmarks_overlay.ply")
                meta = {
                    "id": cam.id, "W": cam.W, "H": cam.H,
                    "K": cam.K.tolist(), "R": cam.R.tolist(), "t": cam.t.tolist(),
                    "head_quat": quat.tolist(), "head_t_cam": t_ch.tolist(),
                    "orientation_deg": [roll, pitch, yaw], "units": "facescape_world"
                }
                (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
                np.save(out_dir / "landmarks_2d.npy", landmarks)   # (68,2) uv
                np.save(out_dir / "landmarks_3d.npy", lm_cam)       # (68,3) camera frame



                


# ---------------------------------------------------------------------------
# Camera input. Both producers below return the SAME type run() consumes:
# a list of Camera dataclasses (id, W, H, K (3x3), R (3x3), t (3,)), all OpenCV
# world->cam. The Camera.__post_init__ arrayifies K/R/t, so the JSON lists below
# get converted for free -- no np.array() needed at construction.
#
# Schema on disk for load_camera(path):
#   [ {"id": "cam0", "W": 512, "H": 512,
#      "K": [[fx,0,cx],[0,fy,cy],[0,0,1]],
#      "R": [[...],[...],[...]], "t": [tx,ty,tz]}, ... ]
# ---------------------------------------------------------------------------
def load_camera(path):
    entries = json.loads(Path(path).read_text())
    return [
        Camera(id=e["id"], W=e["W"], H=e["H"], K=e["K"], R=e["R"], t=e["t"])
        for e in entries
    ]

def default_ring(mesh, n=6, radius=300.0, fov_deg=40.0, W=512, H=512):
    """n look-at cameras evenly spaced in a horizontal ring around the head
    centroid -- a zero-config multi-view set so a bare --id-range renders.
    Returns Camera objects (same type as load_camera)."""
    centroid = mesh.vertices.mean(axis=0)
    K = intrinsics_from_fov(fov_deg, W, H)
    cams = []
    for i in range(n):
        az = 2.0 * np.pi * i / n          # azimuth around +Y (up); i=0 is the front
        eye = centroid + radius * np.array([np.sin(az), 0.0, np.cos(az)])
        R, t = look_at_cv(eye, centroid)       # +Y up; given primitive
        cams.append(Camera(id=f"cam{i:02d}", W=W, H=H, K=K, R=R, t=t))
    return cams


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render FaceScape TU meshes through virtual cameras (RGB + depth + landmarks)."
    )
    parser.add_argument("--id_range", required=True, help='e.g. "801-805" or "801"')
    parser.add_argument("--cameras", help="path to cameras.json")
    parser.add_argument("--orientation", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        metavar=("ROLL", "PITCH", "YAW"),
                        help="head orientation in degrees (0 0 0 = front), Tait-Bryan. "
                             "+yaw=subject's right, +pitch=up, +roll=tilt to subject's right")
    args = parser.parse_args()
    render = ViewRenderer()
    render.run(args.id_range, args.cameras, args.orientation)