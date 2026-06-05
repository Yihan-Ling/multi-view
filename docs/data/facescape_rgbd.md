# FaceScape → virtual multi-view RGB-D

Tooling to turn FaceScape into virtual RGB-D multi-view data with free 3D ground
truth (point clouds + facial landmarks + 6-DoF head pose), plus extraction of the
real captured RGB. See the design memory `facescape-dataset` for why FaceScape.

## Why this is possible

FaceScape ships, per subject/expression: calibrated multi-view cameras
(`params.json`), a raw scan `.ply` (world frame), and a topologically-uniform **TU
model** (canonical frame, shared topology, textured). It does **not** ship depth.
We render the textured TU model through the cameras to synthesize RGB **and** depth,
and read landmarks straight off the known TU topology.

The raw `.ply` is geometry-only (1M verts, **no color**), so synthetic RGB and
landmarks both come from the **TU model**, not the `.ply`.

## Environments (two venvs)

| env | Python | purpose | deps |
|-----|--------|---------|------|
| `.venv`      | 3.14 | model / training | `requirements.txt` |
| `.venv-data` | 3.10 | render / data gen | `requirements-data.txt` |

The render stack (`pyrender`, `trimesh`, …) has no Python 3.14 wheels, and the render
step is offline (writes `png`/`npy` the torch env later reads), so it lives in its own
py3.10 venv. Headless rendering uses EGL: prefix commands with `PYOPENGL_PLATFORM=egl`.

```bash
python3.10 -m venv .venv-data
.venv-data/bin/pip install -r requirements-data.txt
bash scripts/data/setup_toolkit.sh          # clones toolkit @pinned commit + sample
PYOPENGL_PLATFORM=egl .venv-data/bin/python scripts/data/render_rgbd.py --mode demo
```

`third_party/` (the cloned toolkit) is git-ignored and recreated by
`setup_toolkit.sh`.

## Subject selection

`scripts/data/select_subjects.py` — pool is restricted to the login-gated
**`publishable_list`** (only those subjects may appear in paper figures), then
stratified across gender and age bands via `info_list_v2.txt`. Both files come from
the FaceScape download page (TU package). Output: `data/facescape/selection.json`.

## Download (command line, with a checkpoint)

FaceScape is shared via a Google-Drive link issued after license approval. Pull it
with **`rclone`** (preferred) or the deprecated `gdrive` CLI.

- If `Multi-View Data/` exposes **per-subject folders** → fetch only the selected IDs.
- If it is only **ID-range archives** → **STOP and report ranges/sizes**; per the plan
  you decide whether to pull range-by-range and delete unneeded subjects after each.

Then prune to `data/facescape/raw/` with `scripts/data/extract_selection.py`. We also
need the per-subject `Rt_scale_dict` entry (toolkit `predef/`, already cloned) and the
TU model (`<id>/1_neutral.obj` + texture).

## Pipeline

```
select_subjects.py  →  selection.json
        ↓ (download + )
extract_selection.py →  raw/mview/<id>/1_neutral/{params.json,*.jpg}, raw/tu/<id>/1_neutral.obj
        ↓
render_rgbd.py --mode dataset  →  rgbd/<id>/...      (synthetic RGB-D + 3D bundle)
extract_rgb.py                 →  raw_rgb/<id>/...    (real RGB, all valid views)
```

`render_rgbd.py` per subject: loads the TU model, maps canonical→world via the inverse
`Rt_scale`, samples `--n-views` cameras (default 5) spread in azimuth, and per view
renders RGB+depth, back-projects depth to a world point cloud, attaches the 68
landmarks (with occlusion-aware visibility), and computes the 6-DoF head pose.

`--mode demo` runs the whole bundle on the bundled sample TU model with generated ring
cameras (no license needed) — used to validate the pipeline.

## Output schema

```
data/facescape/rgbd/<id>/
  <view>/
    rgb.png            8-bit sRGB                          (train + display)
    depth.npy          float32, FaceScape world units      (train)
    mask.png           valid/face mask                     (train + display)
    cloud_cam.npy      N×6 (xyz + rgb), CAMERA (CV) frame   (per-view 3D, sensor-faithful)
    landmarks_cam.npy  68×3 xyz, CAMERA (CV) frame
    meta.json          K, Rt, P=K[R|t], bbox, units, frames, 6-DoF pose, landmark visibility+uv
    face.ply landmark.ply face-landmark.ply                (display, camera-facing)
    depth_vis.png panel.png lmk_overlay.png                (display, 2D)
  landmarks_world.npy  68×3, shared WORLD frame (fusion target)
  panel.ply            all views' face-landmark clouds side by side, each camera-facing
  panel.png            orthographic front-view render of panel.ply (quick glance)
  tuple_index.json     groups the views as one multi-view sample
```

**Per-view frame.** The per-view 3D arrays are stored in the **camera (CV) frame**
($x_{cam}=Rx_{world}+t$) — what a real RGB-D sensor outputs. The shared **world**-frame
landmarks (`landmarks_world.npy`, identical across views) are kept for multi-view fusion;
camera↔world is lossless via `Rt`, so either frame is recoverable. The display `.ply`
files use the camera frame with a Y/Z flip (`diag(1,-1,-1)`) so a viewer (MeshLab) opens
each face upright and facing front from that camera's direction; `face-landmark.ply`
merges the colored cloud with one **red dot per landmark**, and `panel.ply` lays all
views out side by side.

**Format rationale (display vs. train).** RGB and depth are kept **separate and
lossless** (`rgb.png` + float32 `depth.npy`) so a vanilla MVGFormer dataloader can read
`rgb.png` alone, while an RGBD-modified one also reads `depth.npy` and stacks
$[R,G,B,D] \to (4,H,W)$. `meta.json` carries `P=K[R|t]` for projective attention and
`bbox` for cropping to the model's $256\times256$.

## Conventions & caveats

- **Camera**: CV convention; `Rt` is world→camera $[R\,|\,t]$, $x_{cam}=Rx_{world}+t$.
- **Alignment**: `Rt_scale_dict[id][exp] = [s, R_{cw}|t_{cw}]` maps world→canonical
  ($x_{can}=s\,R_{cw}x_{world}+t_{cw}$); we use the inverse
  $x_{world}=\tfrac{1}{s}R_{cw}^{\top}(x_{can}-t_{cw})$. The toolkit notes scan↔TU
  alignment has "minor misalignment", so landmark GT is accurate to that tolerance.
- **Display `.ply` frame**: per-view `.ply` are in the camera frame with $\mathrm{diag}(1,-1,-1)$
  applied (CV→viewer), so opening one shows the face upright, facing the viewer, from that
  view's camera direction. The `.npy` arrays stay in the raw camera (CV) frame.
- **Depth units** are FaceScape world units (not calibrated to meters); recorded in
  `meta.json`. Rescale later if a metric scale is established.
- **Landmark visibility** is an occlusion test: $|z_{cam}^{lm}-\text{depth}(u,v)|<2\%$.
- Synthetic RGB uses the TU base mesh + 4K texture (no displacement detail; untextured
  eyeballs render white — expected). The `.mtl` ships `Kd=0`; we reset diffuse to white
  or the texture renders black.

## Scaling up

Current pilot: ~10 publishable subjects, neutral only, 5 rendered views. To grow:
add IDs to `selection.json`, raise `--n-views`, add expressions (`--exp`, drop the
neutral-only assumption), and re-run. Point clouds dominate size — tune
`--cloud-stride`.

## Verification (STOP-and-eyeball checkpoints)

- `scripts/data/_selftest_render.py` — single-view render→cloud→landmark on the sample;
  asserts depth back-projection round-trips to source pixels at ~0 px and all 68
  landmarks project in-frame. Opens `data/facescape/_selftest/panel.png`.
- `render_rgbd.py --mode demo` — multi-view bundle; open `panel.png` (or `panel.ply` in
  MeshLab): the 5 camera-facing faces sit upright in a row with red landmark dots on each.
  `cloud_cam`→world via `Rt` matches the shared `landmarks_world.npy`.
