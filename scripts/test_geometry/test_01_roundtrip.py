"""Phase 0 gate: project -> triangulate must recover the original 3D point.

Take known world points, project them into every valid FaceScape view with
`project`, then feed the multi-view 2D observations back through
`triangulate_dlt`. If the recovered 3D points match the originals, the two
geometry primitives are mutually consistent and Phase 0 is closed.
"""

import _init_paths  # noqa: F401

import sys

import numpy as np
import torch

from multi_view.data import facescape_reader
from multi_view.geometry import project, triangulate_dlt

PARAMS = "third_party/facescape_toolkit/samples/sample_mview_data/4_anger/params.json"
N_POINTS = 50
TOL = 1e-6  # mm


def main() -> None:
    rng = np.random.default_rng(0)
    params = facescape_reader.load_params(PARAMS)
    vids = facescape_reader.view_ids(params, valid_only=True)

    # Build the (V, 3, 4) stack of projection matrices, and keep views around.
    Ps = []
    for vid in vids:
        view = facescape_reader.get_view(params, vid)
        Ps.append(view.K @ view.Rt)
    Ps = torch.tensor(np.stack(Ps), dtype=torch.float64)  # (V, 3, 4)
    V = Ps.shape[0]

    # Known world points near the face volume (mm). Origin-ish, modest spread.
    pts_world = torch.tensor(
        rng.uniform(-100.0, 100.0, size=(N_POINTS, 3)), dtype=torch.float64
    )

    max_diff = 0.0
    for i in range(N_POINTS):
        Xw = pts_world[i : i + 1]  # (1, 3)

        # Project this single point into every view -> (V, 2) observations.
        obs = torch.empty((V, 2), dtype=torch.float64)
        for v in range(V):
            obs[v] = project(Xw, Ps[v])[0]

        # Triangulate back from all V views.
        X_rec = triangulate_dlt(obs, Ps)  # (3,)

        diff = (X_rec - pts_world[i]).abs().max().item()
        max_diff = max(max_diff, diff)

    print(f"views: {V}  points: {N_POINTS}")
    print(f"max |X_recovered - X_true| = {max_diff:.3e} mm  (tol {TOL:.0e})")
    if max_diff < TOL:
        print("test_01_roundtrip: PASS")
    else:
        print("test_01_roundtrip: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
