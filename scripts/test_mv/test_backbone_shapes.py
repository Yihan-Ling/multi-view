"""Phase 3 gate: feed one real multi-view sample through MultiViewBackbone and
check the feature-map shape + finiteness.

Run:  .venv/bin/python scripts/test_mv/test_backbone_shapes.py
"""

import _init_paths  # noqa: F401
import os

import torch

from _init_paths import REPO_ROOT
from multi_view.data.facescape_dataset import MultiViewFaceScape
from multi_view.backbone import MultiViewBackbone

ROOT = REPO_ROOT / "data" / "facescape" / "virtual_camera_data"


def main() -> None:
    subs = sorted((d for d in os.listdir(ROOT) if d.isdigit()), key=int)[:1]
    ds = MultiViewFaceScape(ROOT, subs)
    sample = ds[0]
    rgbd = sample["rgbd"].unsqueeze(0)          # (1, N, 4, H, W)
    B, N, C, H, W = rgbd.shape
    print(f"input rgbd: {tuple(rgbd.shape)}")

    model = MultiViewBackbone(pretrained=False).eval()
    with torch.no_grad():
        feats = model(rgbd)

    exp = (B, N, model.out_channels, H // 4, W // 4)
    print(f"output feats: {tuple(feats.shape)}  expected {exp}")
    assert feats.shape == exp, "feature-map shape mismatch"
    assert torch.isfinite(feats).all(), "non-finite values in feature map"
    print("PHASE 3 GATE PASSED: shapes correct, values finite.")


if __name__ == "__main__":
    main()
