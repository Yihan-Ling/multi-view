"""Phase 6 - deep-supervised losses for the multi-view 3D-landmark decoder.

Two terms, summed over ALL decoder layers (auxiliary/deep supervision, the DETR
trick that stabilizes training):
  - 3D landmark L1 between each layer's predicted 3D and the GT.
  - per-view 2D L1 between each layer's refined pixel landmarks and the GT pixels,
    MASKED by visibility (the 2D is what triangulation consumes, so only supervise
    views where the landmark is actually seen).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_l1_2d(pred: torch.Tensor, gt: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
    """pred, gt: (B,N,Q,2); vis: (B,N,Q). Mean L1 over visible entries only."""
    m = vis.unsqueeze(-1)                       # (B,N,Q,1)
    diff = (pred - gt).abs() * m
    return diff.sum() / (m.sum() * 2 + 1e-6)    # *2: two coords per landmark


def decoder_losses(preds_3d, preds_2d, gt_3d, gt_2d, vis, lambda_2d: float = 0.1):
    """preds_3d: list[(B,Q,3)], preds_2d: list[(B,N,Q,2)] (one per decoder layer).
    Returns a dict with the total and each component (for logging)."""
    loss_3d = preds_3d[0].new_zeros(())
    loss_2d = preds_3d[0].new_zeros(())
    for p3, p2 in zip(preds_3d, preds_2d):
        loss_3d = loss_3d + F.l1_loss(p3, gt_3d)
        loss_2d = loss_2d + masked_l1_2d(p2, gt_2d, vis)
    total = loss_3d + lambda_2d * loss_2d
    return {"total": total, "loss_3d": loss_3d, "loss_2d": loss_2d}


@torch.no_grad()
def mpjpe_mm(pred_3d: torch.Tensor, gt_3d: torch.Tensor) -> float:
    """Mean per-joint position error (mm): mean L2 distance over landmarks/batch."""
    return (pred_3d - gt_3d).norm(dim=-1).mean().item()
