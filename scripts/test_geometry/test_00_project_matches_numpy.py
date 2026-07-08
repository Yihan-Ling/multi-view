"""Phase 0 checkpoint: torch `project` must match numpy `project_points`.

`multi_view/geometry.py::project` and `multi_view/data/facescape_reader.py::project_points`
were written independently. If they agree to ~1e-6 on real calibrated cameras, the
torch projection convention (P = K@Rt, homogenize, perspective divide) is proven
correct and everything downstream can trust it.
"""

import _init_paths  # noqa: F401

import sys

import numpy as np
import torch

from multi_view.data import facescape_reader
from multi_view.geometry import project

PARAMS = "third_party/facescape_toolkit/samples/sample_mview_data/4_anger/params.json"
N_POINTS = 100
TOL = 1e-6


def main() -> None:
    rng = np.random.default_rng(0)
    params = facescape_reader.load_params(PARAMS)
    vids = facescape_reader.view_ids(params, valid_only=True)

    max_diff = 0.0
    for vid in vids:
        view = facescape_reader.get_view(params, vid)
        K, Rt = view.K, view.Rt  # (3,3), (3,4) float64

        # Build N test points *in front of* this camera: sample positive depths
        # in the camera frame, then map out to the world frame so both functions
        # receive world coordinates (and no point sits near zero depth).
        z_cam = rng.uniform(200.0, 1000.0, size=(N_POINTS, 1))          # mm depth
        xy_cam = rng.uniform(-300.0, 300.0, size=(N_POINTS, 2))
        cam = np.concatenate([xy_cam, z_cam], axis=1)                   # (N,3) camera frame
        R, t = Rt[:, :3], Rt[:, 3]
        pts_world = (cam - t) @ R                                       # R^T @ (cam - t)

        # numpy reference
        uv_np = facescape_reader.project_points(K, Rt, pts_world)             # (N,2)

        # torch version under test (float64 to match numpy precision)
        P = K @ Rt                                                     # (3,4)
        uv_torch = project(
            torch.tensor(pts_world, dtype=torch.float64),
            torch.tensor(P, dtype=torch.float64),
        ).numpy()

        diff = np.abs(uv_torch - uv_np).max()
        max_diff = max(max_diff, diff)

    print(f"views tested: {len(vids)}  points/view: {N_POINTS}")
    print(f"max |uv_torch - uv_numpy| = {max_diff:.3e}  (tol {TOL:.0e})")
    if max_diff < TOL:
        print("test_00_project_matches_numpy: PASS")
    else:
        print("test_00_project_matches_numpy: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
