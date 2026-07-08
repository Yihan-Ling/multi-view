"""Phase 2 - build the decoder's query-init template from the FaceScape bilinear model.

The decoder starts every sample from a fixed template of 68 3D landmark
positions (`ref_3d`) and refines it over 4 layers. We take that template's
SHAPE from the FaceScape bilinear model's mean face (average of 847 subjects at
neutral expression) and its LOCATION/scale (the query box) from where faces
actually appear in our render set (heads are randomly posed in the world frame,
so the model's canonical origin is not where our faces sit).

    shape   <- bilinear model:  core . id_mean . neutral_exp  -> slice 68 landmarks
    box     <- dataset spread:   SPACE_CENTER + SPACE_SIZE covering all subjects

Outputs (under multi_view/assets/):
    mean_face_68.npy   (68, 3)  template shape, centered on its own centroid
    query_space.json   SPACE_CENTER (3,) world location + SPACE_SIZE (mm) cube

Gate (eyeball): scratch/phase2_mean_face.png -- a recognisable face, and the
printed bilinear-vs-dataset residual should be small (~single-digit mm).

Run:  .venv/bin/python scripts/mean_face/build_mean_face.py
"""

from __future__ import annotations

import json

import _init_paths  # noqa: F401
import numpy as np

from _init_paths import REPO_ROOT

MODEL = (REPO_ROOT / "data" / "facescape" / "bilinear_model_v1_6"
         / "facescape_bm_v1.6_847_50_52_id_front.npz")
ROOT = REPO_ROOT / "data" / "facescape" / "virtual_camera_data"
ASSETS = REPO_ROOT / "multi_view" / "assets"
SCRATCH = REPO_ROOT / "scratch"

BOX_MARGIN = 1.15  # pad the dataset bbox so every landmark sits comfortably inside


# ----------------------------------------------------------------------------- #
# Template SHAPE from the bilinear model (mechanical model eval).
# ----------------------------------------------------------------------------- #
def bilinear_mean_face(model_path) -> np.ndarray:
    """Average neutral face -> 68 landmarks, centered on its own centroid."""
    d = np.load(model_path, allow_pickle=True)
    core = d["shape_bm_core"].astype(np.float64)          # (78834, 52, 50)
    # residual conversion, mirroring facescape_bm.__init__ (channel 0 = neutral
    # basis; a no-op for the neutral eval below, kept for fidelity).
    sub = np.stack((core[:, 0, :],) * core.shape[1], axis=1)
    res = core - sub
    res[:, 0, :] = core[:, 0, :]
    core = res

    id_vec = d["id_mean"].astype(np.float64)              # (50,) mean identity
    exp = np.zeros(core.shape[1]); exp[0] = 1.0           # neutral expression
    verts = core.dot(id_vec).dot(exp).reshape(-1, 3)      # (26278, 3) model frame
    lm = verts[d["lm_list_v16"]]                          # (68, 3)
    return lm - lm.mean(axis=0)                           # center on centroid


# ----------------------------------------------------------------------------- #
# Query BOX from the dataset (where faces appear in the world frame).
# ----------------------------------------------------------------------------- #
def load_subject_world(subj_dir) -> np.ndarray:
    """Lift a subject's landmarks to world via view 0 (verified consistent)."""
    lm_cam = np.load(subj_dir / "0" / "landmarks_3d.npy").astype(np.float64)
    meta = json.loads((subj_dir / "0" / "meta.json").read_text())
    R = np.asarray(meta["R"], dtype=np.float64)
    t = np.asarray(meta["t"], dtype=np.float64).reshape(3)
    return (lm_cam - t) @ R


def discover_subjects(root):
    return sorted((d for d in root.iterdir() if d.is_dir() and d.name.isdigit()),
                  key=lambda d: int(d.name))


def dataset_query_box(subject_worlds: np.ndarray):
    """subject_worlds: (S, 68, 3). Return (SPACE_CENTER (3,), SPACE_SIZE float)."""
    pts = subject_worlds.reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = (lo + hi) / 2.0
    size = float((hi - lo).max() * BOX_MARGIN)            # a cube that covers all
    return center, size


# ----------------------------------------------------------------------------- #
# Validation: is the model mean representative of our subjects?
# ----------------------------------------------------------------------------- #
def kabsch_residual(A: np.ndarray, B: np.ndarray) -> float:
    """Mean per-landmark distance after rigidly aligning A (68,3) onto B (68,3)."""
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    U, S, Vt = np.linalg.svd(Ac.T @ Bc)
    Dg = np.eye(3); Dg[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    Rk = Vt.T @ Dg @ U.T
    aligned = (Rk @ Ac.T).T
    return float(np.sqrt(((aligned - Bc) ** 2).sum(1)).mean())


def main() -> None:
    mean_face = bilinear_mean_face(MODEL)                 # (68,3) model shape

    subs = discover_subjects(ROOT)
    worlds = np.stack([load_subject_world(s) for s in subs])  # (S,68,3)
    center, size = dataset_query_box(worlds)

    ASSETS.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    np.save(ASSETS / "mean_face_68.npy", mean_face.astype(np.float32))
    (ASSETS / "query_space.json").write_text(json.dumps(
        {"SPACE_CENTER": center.tolist(), "SPACE_SIZE": size,
         "source": "bilinear_v1.6_id_front mean; box from virtual_camera_data"},
        indent=2))

    resid = np.array([kabsch_residual(mean_face, w) for w in worlds])
    print(f"loaded {len(subs)} subjects for the box")
    print("mean face extent (mm):", (mean_face.max(0) - mean_face.min(0)).round(1))
    print("SPACE_CENTER:", center.round(1), " SPACE_SIZE (mm):", round(size, 1))
    print(f"bilinear-vs-dataset residual (mm): mean {resid.mean():.2f}  "
          f"min {resid.min():.2f}  max {resid.max():.2f}")
    print("saved ->", ASSETS / "mean_face_68.npy", "and query_space.json")

    # --- eyeball gate: front (XY) + side (ZY) scatter of the template ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(mean_face[:, 0], mean_face[:, 1], mean_face[:, 2], s=12)
    ax.set_title("3D")
    for k, (i, j, name) in enumerate([(0, 1, "front XY"), (2, 1, "side ZY")], start=2):
        a = fig.add_subplot(1, 3, k)
        a.scatter(mean_face[:, i], mean_face[:, j], s=12)
        for n in range(68):
            a.annotate(str(n), (mean_face[n, i], mean_face[n, j]), fontsize=5)
        a.set_aspect("equal"); a.set_title(name)  # +Y is up in model frame
    fig.tight_layout()
    out = SCRATCH / "phase2_mean_face.png"
    fig.savefig(out, dpi=130)
    print("gate image ->", out)


if __name__ == "__main__":
    main()
