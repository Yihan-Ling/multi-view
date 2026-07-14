import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import cv2
import trimesh
import pyrender          # run with PYOPENGL_PLATFORM=egl for headless GPU
from PIL import Image
from tqdm import tqdm

# Repo root on the path so we can reuse the training-time augmentor (single source
# of truth for the bg+photometric recipe) when baking it into the renders.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

@dataclass
class Camera:
    id: str
    W: int
    H: int
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray

    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=float)
        self.R = np.asarray(self.R, dtype=float)
        self.t = np.asarray(self.t, dtype=float)
        
@dataclass
class Light:
    intensity: float
    ambient: float
    direction: np.ndarray
    
    def __post_init__(self):
        self.direction = np.asarray(self.direction, dtype=float)


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


def light_pose_from_direction(direction):
    """World pose for a pyrender DirectionalLight that travels along `direction`.

    A DirectionalLight emits along the LOCAL -Z axis of its node pose, so we
    build a pose whose local -Z points the way the light travels. Only the
    direction matters for a directional light (it has no position), so we pick
    any stable perpendicular frame; the up-vector swap handles the degenerate
    case where the light is nearly vertical and `cross(up, z)` would collapse.
    """
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    z = -d                                       # local +Z is opposite the travel dir
    up = np.array([0.0, 1.0, 0.0])
    if abs(z @ up) > 0.99:                        # light almost vertical -> new up
        up = np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([x, y, z], axis=1)   # columns = light axes in world
    return pose


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
    
    def load_mesh(self, id, exp="1_neutral"):
        id_range_folder = self.find_id_range_folder(id)
        obj = self.data_root / id_range_folder / id / f"{exp}.obj"
        mesh = trimesh.load(obj, process=False)   # process=False keeps vertex order
        if isinstance(mesh, trimesh.Scene):       # rare, but guard it
            mesh = mesh.dump(concatenate=True)

        mat = mesh.visual.material
        if hasattr(mat, "baseColorFactor"):
            mat.baseColorFactor = np.array([255, 255, 255, 255], dtype=np.uint8)
        if hasattr(mat, "diffuse"):
            mat.diffuse = np.array([255, 255, 255, 255], dtype=np.uint8)

        raw = np.array([ln.split()[1:4] for ln in obj.read_text().splitlines()
                        if ln.startswith("v ")], dtype=float)   # (26317, 3), file order
        mesh.metadata["lm_world"] = raw[self.lm_idx]            # (68, 3)
        return mesh

 
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
        mesh.metadata["head_R"] = R            # head->world rotation
        mesh.metadata["head_t"] = pivot        # head origin (centroid) in world
        return mesh

    # ---- helper: OpenCV (R,t) -> pyrender's OpenGL pose -----------------
    def _gl_pose(self, R, t):
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t
        return c2w @ np.diag([1.0, -1.0, -1.0, 1.0])

    def render(self, mesh, cam, light: Light = None):
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[light.ambient if light else 0.5]*3)
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
        pcam = pyrender.IntrinsicsCamera(fx=cam.K[0, 0], fy=cam.K[1, 1],
                                         cx=cam.K[0, 2], cy=cam.K[1, 2],
                                         znear=1.0, zfar=5000.0)
        pose = self._gl_pose(cam.R, cam.t)
        scene.add(pcam, pose=pose)
        
        if light is None:
            scene.add(pyrender.DirectionalLight(intensity=3.0), pose=pose)
        else:
            scene.add(pyrender.DirectionalLight(intensity=light.intensity), pose=light_pose_from_direction(light.direction))

        renderer = pyrender.OffscreenRenderer(cam.W, cam.H)
        color, depth = renderer.render(scene)     # (H,W,3) uint8, (H,W) float32
        renderer.delete()
        return color, depth

    def project_landmarks(self, mesh, cam):
        pts = mesh.metadata["lm_world"]   # (68,3) carried from the raw 'v' order
        x_cam = (cam.R@pts.T).T + cam.t
        fx, fy = cam.K[0,0], cam.K[1,1]
        cx, cy = cam.K[0,2], cam.K[1,2]
        x, y, z = x_cam.T
        u = fx*x/z + cx
        v = fy*y/z + cy
        return np.stack([u, v], axis=1)

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

    
    def draw_landmarks(self, img, landmarks):
        new_img = img.copy()
        for landmark in landmarks:
            cv2.circle(img=new_img, center=(int(landmark[0]), int(landmark[1])), radius=2, color=(255,0,0), thickness=-1)
        return new_img
            
    def write_view(self, out_dir, color, depth, landmarks):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        Image.fromarray(color).save(out_dir / "rgb.png")
        np.save(out_dir / "depth.npy", depth)
        landmark_overlay = self.draw_landmarks(color, landmarks)
        Image.fromarray(landmark_overlay).save(out_dir/"rgb_landmark_overlay.png")
        return landmark_overlay

    def make_panel(self, images):
        pil = [Image.fromarray(im) for im in images]
        h = max(im.height for im in pil)
        panel = Image.new("RGB", (sum(im.width for im in pil), h), (0, 0, 0))
        x = 0
        for im in pil:
            panel.paste(im, (x, 0))
            x += im.width
        return panel
    
    def save_panel(self, path, images):
        panel = self.make_panel(images=images)
        panel.save(path)
        
    def save_grid(self, path, grid):
        strips = [self.make_panel(panel) for panel in grid]
        
        W = max(s.width for s in strips)
        H = sum(s.height for s in strips)
        
        grid = Image.new("RGB", (W,H), (0,0,0))
        
        y=0
        
        for strip in strips:
            grid.paste(strip, (0, y))
            y+=strip.height
        
        grid.save(path)
        
        
        

    def run(self, id_range, cameras, orientation=(0.0, 0.0, 0.0), lighting=False,
            rand_pose=False, rand_ring=False, variants=1, augmentor=None, lean=False):
        # parse id_range
        if "-" in id_range:
            l, r = id_range.split("-")
            ids = [str(i) for i in range(int(l), int(r)+1)]
        else:
            ids = [id_range]

        bar = tqdm(ids, desc="rendering")
        for id in bar:
            bar.set_postfix_str(id)
            # some FaceScape subjects (e.g. 832) ship textures but no .obj geometry
            # in the trainset zip -- skip them instead of crashing the whole run.
            obj = self.data_root / self.find_id_range_folder(id) / id / "1_neutral.obj"
            if not obj.is_file():
                print(f"  warning: no mesh for id {id}, skipping: {obj}")
                continue

            # Each variant = a fresh draw of head pose + camera ring + baked RGB aug,
            # written as its own multi-view item "<id>_<k>" (variants==1 keeps the
            # plain "<id>" folder for backward compatibility). orient_head mutates the
            # mesh in place, so reload it per variant.
            for k in range(variants):
                roll, pitch, yaw = random_orientation() if rand_pose else orientation
                mesh = self.load_mesh(id=id)
                self.orient_head(mesh, roll, pitch, yaw)
                lm_w = mesh.metadata["lm_world"]    # landmark in world frame
                if cameras:
                    cams = load_camera(cameras)
                elif rand_ring:
                    cams = random_ring(mesh)       # resampled per variant
                else:
                    cams = default_ring(mesh)
                subj_out = id if variants == 1 else f"{id}_{k}"
                overlays = []                       # per-cam overlays for the panel

                for i, cam in enumerate(cams):
                    cam.id = str(i)                 # output ids are ALWAYS 0,1,2,...
                    color, depth = self.render(mesh, cam)
                    # Bake domain randomization (bg composite + HRNet photometric) into
                    # the saved RGB, keyed off the raw depth>0 mask so bg leaks through
                    # the eye holes. Fresh randomness per view; depth + GT stay clean.
                    if augmentor is not None:
                        rgb01 = augmentor.apply(color.astype(np.float32) / 255.0,
                                                depth > 0, np.random.default_rng())
                        color = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
                    landmarks = self.project_landmarks(mesh, cam)
                    lm_cam = (cam.R @ lm_w.T).T + cam.t     # landmark in camera frame
                    R_ch   = cam.R @ mesh.metadata["head_R"]    # head orientation in camera frame
                    t_ch   = cam.R @ mesh.metadata["head_t"] + cam.t    # head position in camera frame
                    quat   = rotmat_to_quat(R_ch)   # R_ch in quaternion

                    out_dir = self.out_root / subj_out / cam.id
                    out_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(color).save(out_dir / "rgb.png")
                    np.save(out_dir / "depth.npy", depth)
                    meta = {
                        "id": cam.id, "W": cam.W, "H": cam.H,
                        "K": cam.K.tolist(), "R": cam.R.tolist(), "t": cam.t.tolist(),
                        "head_quat": quat.tolist(), "head_t_cam": t_ch.tolist(),
                        "orientation_deg": [roll, pitch, yaw], "units": "facescape_world"
                    }
                    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
                    np.save(out_dir / "landmarks_2d.npy", landmarks)   # (68,2) uv
                    np.save(out_dir / "landmarks_3d.npy", lm_cam)       # (68,3) camera frame

                    # Debug artifacts (landmark overlay, point clouds) -- skipped in
                    # --lean mode, which is what you want when baking many variants.
                    if not lean:
                        overlay = self.draw_landmarks(color, landmarks)
                        Image.fromarray(overlay).save(out_dir / "rgb_landmark_overlay.png")
                        overlays.append(overlay)
                        pts, cols = self.backproject(depth=depth, K=cam.K, color=color)
                        flip = np.array([1.0, -1.0, -1.0])
                        pts_gl = pts * flip
                        trimesh.PointCloud(vertices=pts_gl, colors=cols).export(out_dir / "rgbd.ply")
                        lm_dots = np.tile([255, 0, 0], (len(lm_cam), 1))
                        all_pts = np.vstack([pts_gl, lm_cam * flip])
                        all_colors = np.vstack([cols, lm_dots])
                        trimesh.PointCloud(vertices=all_pts, colors=all_colors).export(out_dir / "rgbd_landmarks_overlay.ply")

                if not lean:
                    # all camera overlays side by side, under the item folder
                    self.save_panel(self.out_root / subj_out / "panel.png", overlays)
                    if lighting:
                        grid = []
                        for kk in range(1, 5):
                            row = []
                            light = generate_random_light()
                            for i, cam in enumerate(cams):
                                cam.id = str(i)
                                color, _ = self.render(mesh, cam, light=light)
                                out_dir = self.out_root / subj_out / cam.id
                                Image.fromarray(color).save(out_dir / f"rgb_{kk}.png")
                                row.append(color)
                            grid.append(row)
                        self.save_grid(path=self.out_root / subj_out / "lighting_panel.png", grid=grid)



def load_camera(path):
    entries = json.loads(Path(path).read_text())
    return [
        Camera(id=e["id"], W=e["W"], H=e["H"], K=e["K"], R=e["R"], t=e["t"])
        for e in entries
    ]

def default_ring(mesh, n=5, radius=380.0, fov_deg=40.0, W=512, H=512,
                 az_min_deg=-50.0, az_max_deg=50.0):
    """n look-at cameras spanned evenly in azimuth across [az_min_deg, az_max_deg]
    around the head centroid, centered on the front (az=0). A front-biased arc, not
    a full circle -- the back of the head carries no useful info. Defaults: 5 cams
    from -50deg to +50deg. Returns Camera objects (same type as load_camera)."""
    centroid = mesh.vertices.mean(axis=0)
    K = intrinsics_from_fov(fov_deg, W, H)
    azimuths = np.radians(np.linspace(az_min_deg, az_max_deg, n))   # az=0 is the front
    cams = []
    for i, az in enumerate(azimuths):
        eye = centroid + radius * np.array([np.sin(az), 0.0, np.cos(az)])
        R, t = look_at_cv(eye, centroid)       # +Y up; given primitive
        cams.append(Camera(id=f"cam{i:02d}", W=W, H=H, K=K, R=R, t=t))
    return cams

def random_ring(mesh, n=5, W=512, H=512):
    """Randomized front-biased camera ring, resampled PER SUBJECT (iteration-2
    robustness). Removes the fixed-rig limitation of `default_ring`: the model no
    longer sees the same 5 viewpoints every subject. n is kept fixed (=5) so all
    subjects have the same view count (the multi-view model batches on N); only the
    ring GEOMETRY varies -- radius, FOV, azimuth arc (center + width + per-cam
    jitter), and ring elevation (center + per-cam jitter). Still front-biased (the
    back of the head carries no info). Returns Camera objects like default_ring."""
    centroid = mesh.vertices.mean(axis=0)
    radius = np.random.uniform(340.0, 420.0)
    fov_deg = np.random.uniform(35.0, 45.0)
    K = intrinsics_from_fov(fov_deg, W, H)
    # azimuth: random arc center + half-width, plus small per-camera jitter.
    az_center = np.random.uniform(-15.0, 15.0)
    az_half = np.random.uniform(35.0, 60.0)
    az_deg = np.linspace(az_center - az_half, az_center + az_half, n) \
        + np.random.uniform(-5.0, 5.0, size=n)
    # elevation: random ring tilt + per-camera jitter (vertical viewpoint diversity).
    el_deg = np.random.uniform(-12.0, 12.0) + np.random.uniform(-5.0, 5.0, size=n)
    azr, elr = np.radians(az_deg), np.radians(el_deg)
    cams = []
    for i, (az, el) in enumerate(zip(azr, elr)):
        direction = np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
        eye = centroid + radius * direction
        R, t = look_at_cv(eye, centroid)       # +Y up; given primitive
        cams.append(Camera(id=f"cam{i:02d}", W=W, H=H, K=K, R=R, t=t))
    return cams


def random_orientation(pitch_max=25.0, yaw_max=35.0):
    """Random head orientation for --rand_pose. Samples pitch + yaw -- the
    out-of-plane axes that train-time augmentation CANNOT synthesize (the fixed
    camera ring renders pitch=0 only and 5 discrete yaws). Roll is left at 0
    because HRNet's ROT_FACTOR augmentation already covers in-plane roll, and
    the camera ring's azimuth compounds with this yaw for continuous coverage."""
    pitch = np.random.uniform(-pitch_max, pitch_max)
    yaw   = np.random.uniform(-yaw_max, yaw_max)
    return (0.0, pitch, yaw)


def generate_random_light() -> Light:
    x = np.random.uniform(-1.0, 1.0)    # left/right: full swing to either cheek
    y = np.random.uniform(-0.75, 0.5)    # up/down: gentler, avoids harsh top/bottom light
    z = np.random.uniform(-1.0, 0.3)   # ALWAYS negative -> light comes from the front
    
    intensity = np.random.uniform(5.0, 10.0)
    ambient = np.random.uniform(0.2, 0.5)
    return Light(intensity=intensity, ambient=ambient, direction=(x,y,z))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render FaceScape TU meshes through virtual cameras (RGB + depth + landmarks)."
    )
    parser.add_argument("--id_range", required=True, help='e.g. "801-805" or "801"')
    parser.add_argument("--cameras", help="path to cameras.json")
    parser.add_argument("--orientation", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("ROLL", "PITCH", "YAW"))
    parser.add_argument("--lighting", action="store_true", help="True for random lighting conditions, default false")
    parser.add_argument("--rand_pose", action="store_true", help="randomize head pitch+yaw per subject (overrides --orientation)")
    parser.add_argument("--rand_ring", action="store_true", help="randomize the camera ring geometry per subject (iteration-2 robustness)")
    parser.add_argument("--variants", type=int, default=1,
                        help="baked augmentation variants per subject; each is a fresh "
                             "pose+ring+RGB-aug draw written as <id>_<k> (K-variant bake)")
    parser.add_argument("--bg_dir", default="data/backgrounds/indoor/Images",
                        help="background image pool for baked bg compositing")
    parser.add_argument("--bg_prob", type=float, default=0.0,
                        help="per-view prob of baking a random background (0=off)")
    parser.add_argument("--photometric", action="store_true",
                        help="bake HRNet's photometric ISP jitter into the saved RGB")
    parser.add_argument("--lean", action="store_true",
                        help="skip debug artifacts (overlays, point clouds, panels); "
                             "use when baking many variants to save disk")
    parser.add_argument("--data_root", default="data/facescape", help="root holding the TU meshes")
    parser.add_argument("--out_root", default="data/facescape/virtual_camera_data",
                        help="output dir; use a NEW dir for a random-ring set to keep the fixed-ring set")
    args = parser.parse_args()

    # Build the shared bg+photometric augmentor if baking is requested.
    from multi_view.data.augment import AugConfig, MultiViewAugmentor
    aug_cfg = AugConfig(bg_dir=args.bg_dir, bg_prob=args.bg_prob,
                        photometric=args.photometric)
    augmentor = MultiViewAugmentor(aug_cfg) if aug_cfg.enabled else None

    render = ViewRenderer(data_root=args.data_root, out_root=args.out_root)
    render.run(args.id_range, args.cameras, args.orientation, lighting=args.lighting,
               rand_pose=args.rand_pose, rand_ring=args.rand_ring,
               variants=args.variants, augmentor=augmentor, lean=args.lean)