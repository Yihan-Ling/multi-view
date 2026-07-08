"""Phase 4 gate: a query placed at a known GT landmark's 3D position must sample
at that landmark's GT 2D pixel (reuses the dataset's stored landmarks_2d).

Parameter-free: uses RANDOM feature maps (no backbone), so this checks only the
projection + grid geometry, not any learned weights.

Run:  .venv/bin/python scripts/test_mv/test_proj_attn.py
"""

import _init_paths  # noqa: F401
import os

import torch

from _init_paths import REPO_ROOT
from multi_view.data.facescape_dataset import MultiViewFaceScape
from multi_view.decoder import ProjectiveAttention

ROOT = REPO_ROOT / "data" / "facescape" / "virtual_camera_data"


def main() -> None:
    subs = sorted((d for d in os.listdir(ROOT) if d.isdigit()), key=int)[:1]
    ds = MultiViewFaceScape(ROOT, subs)
    s = ds[0]

    query_3d = s["landmarks_3d"].unsqueeze(0)          # (1, 68, 3) world GT
    proj = s["proj"].unsqueeze(0)                      # (1, N, 3, 4)
    gt_uv = s["landmarks_2d"].unsqueeze(0)             # (1, N, 68, 2) GT pixels
    B, N = proj.shape[0], proj.shape[1]
    C, Hf, Wf = 256, 128, 128
    H_img = W_img = s["rgbd"].shape[-1]                # 512

    feat_maps = torch.randn(B, N, C, Hf, Wf)
    attn = ProjectiveAttention(d_model=C)
    feat, sampled, uv = attn(query_3d, feat_maps, proj, image_hw=(H_img, W_img))

    print(f"feat: {tuple(feat.shape)}   sampled: {tuple(sampled.shape)}   uv: {tuple(uv.shape)}")
    assert feat.shape == (B, 68, C), "aggregated feature shape wrong"
    assert torch.isfinite(feat).all(), "non-finite features"

    err = (uv - gt_uv).abs().max().item()
    print(f"max |projected uv - GT landmark pixel| = {err:.4f} px")
    assert err < 1.0, "queries at GT landmarks do not project to GT pixels"
    print("PHASE 4 GATE PASSED: queries project to their GT pixels; shapes OK.")


if __name__ == "__main__":
    main()
