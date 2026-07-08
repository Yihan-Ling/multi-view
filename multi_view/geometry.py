"""Differentiable multi-view geometry primitives (torch).

* CV camera, world->camera extrinsics ``x_cam = R @ x_world + t``.
* Projection matrix ``P = K @ Rt`` (3x4); pixel = (P @ [X;1]) with xy divided by
  the 3rd (depth) coordinate.
"""

from __future__ import annotations

import torch


def project(points_3d: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Project 3D world points into 2D pixels through a calibrated camera.

    Args:
        points_3d: (N, 3) world-frame points.
        P:         (3, 4) projection matrix P = K @ Rt.

    Returns:
        (N, 2) pixel coordinates (u, v), after the perspective divide.
    """
    ones = torch.ones((points_3d.shape[0], 1), dtype=points_3d.dtype, device=points_3d.device)
    points_h = torch.cat((points_3d, ones), dim=-1)
    
    proj = points_h @ P.T

    uv = proj[:, :2] / proj[:, 2:3]
    return uv


def triangulate_dlt(
    points_2d: torch.Tensor,
    Ps: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Recover one 3D point from its 2D observations in V calibrated views (DLT).

    Args:
        points_2d: (V, 2) pixel observations of the same world point.
        Ps:        (V, 3, 4) projection matrices P = K @ Rt, one per view.
        weights:   (V,) optional per-view confidence. Each view's two DLT rows are
                   scaled by its weight, so low-confidence views pull the solution
                   less. None = all views equal (identical to the unweighted DLT).

    Returns:
        (3,) world-frame 3D point.
    """
    # p1, p2, p3: the three rows of every view's P.  each (V, 4).
    p1 = Ps[:, 0, :]
    p2 = Ps[:, 1, :]
    p3 = Ps[:, 2, :]

    u = points_2d[:, 0:1]   # (V, 1) keep trailing dim so it broadcasts over the 4 cols
    v = points_2d[:, 1:2]   # (V, 1)

    # two DLT rows per view:  (u*p3 - p1) . Xh = 0 ,  (v*p3 - p2) . Xh = 0
    row_u = u*p3 - p1             # (V, 4)
    row_v = v*p3 - p2             # (V, 4)

    # confidence weighting: scale each view's two rows by its weight (V,1) so a
    # downweighted view contributes less to the least-squares null-space solve.
    if weights is not None:
        w = weights[:, None]     # (V, 1)
        row_u = row_u * w
        row_v = row_v * w

    # stack into A of shape (2V, 4)
    A = torch.cat([row_u, row_v], dim=0)

    # smallest-singular-vector of A -> homogeneous Xh (length 4)
    U, S, Vh = torch.linalg.svd(A)
    Xh = Vh[-1]

    # dehomogenize
    X = Xh[:3]/Xh[3]
    return X


def triangulate_dlt_batch(
    points_2d: torch.Tensor,
    Ps: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched confidence-weighted DLT: triangulate M points at once.

    Same math as triangulate_dlt, vectorized over a leading M axis (e.g. M = B*Q
    landmarks) so the decoder avoids a Python loop of SVDs.

    Args:
        points_2d: (M, V, 2) each point's V pixel observations.
        Ps:        (M, V, 3, 4) projection matrices per point per view.
        weights:   (M, V) optional per-view confidence (scales that view's rows).

    Returns:
        (M, 3) triangulated world points.
    """
    p1 = Ps[..., 0, :]                     # (M, V, 4)
    p2 = Ps[..., 1, :]
    p3 = Ps[..., 2, :]

    u = points_2d[..., 0:1]                # (M, V, 1)
    v = points_2d[..., 1:2]

    row_u = u * p3 - p1                     # (M, V, 4)
    row_v = v * p3 - p2

    if weights is not None:
        w = weights[..., None]             # (M, V, 1)
        row_u = row_u * w
        row_v = row_v * w

    A = torch.cat([row_u, row_v], dim=1)   # (M, 2V, 4)
    U, S, Vh = torch.linalg.svd(A)         # Vh: (M, 4, 4)
    Xh = Vh[:, -1, :]                      # (M, 4) smallest right-singular vector
    X = Xh[:, :3] / Xh[:, 3:4]             # (M, 3) dehomogenize
    return X
