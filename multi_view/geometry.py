"""Differentiable multi-view geometry primitives (torch).

Torch counterparts to the numpy helpers in ``data/facescape.py``. These sit on
the model path (autograd flows through them), so they must reproduce the numpy
conventions exactly:

* CV camera, world->camera extrinsics ``x_cam = R @ x_world + t``.
* Projection matrix ``P = K @ Rt`` (3x4); pixel = (P @ [X;1]) with xy divided by
  the 3rd (depth) coordinate.
"""

from __future__ import annotations

import torch


def project(points_3d: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Project world points to pixels.

    points_3d: (N, 3) world points
    P:         (3, 4) projection matrix K[R|t]
    returns:   (N, 2) pixel coords
    """
    # a. homogeneous points (N, 4): append a column of ones
    #    (match points_3d.dtype and .device)
    ones = ...
    points_h = ...

    # b. multiply: (N, 4) @ (4, 3) -> (N, 3)
    proj = ...

    # c. divide xy by the depth (3rd column) -> (N, 2)
    #    use proj[:, 2:3] (keepdim) so it broadcasts against proj[:, :2]
    uv = ...
    return uv
