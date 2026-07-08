"""Phase 7 - the full early-fusion multi-view 3D-landmark model.

    rgbd (B,N,4,H,W) -> MultiViewBackbone -> feats (B,N,256,h,w)
                     -> MultiViewDecoder  -> per-layer 3D + 2D landmark predictions

Two knobs for the experiment:
  - use_depth: False zeroes the depth channel -> the RGB-only arm of the ablation.
  - img_size:  optional bilinear resize of the input FED TO THE BACKBONE (for
               speed/memory). Geometry is untouched: proj matrices and the 2D GT
               stay in the ORIGINAL pixel coords, and grid_sample normalizes by the
               original image size (passed as image_hw), which is resolution-free.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from multi_view.backbone import MultiViewBackbone
from multi_view.decoder import MultiViewDecoder


class MultiViewLandmark3D(nn.Module):
    def __init__(self, assets_dir, num_layers: int = 4, pretrained: bool = False,
                 use_depth: bool = True, img_size: int | None = None) -> None:
        super().__init__()
        self.backbone = MultiViewBackbone(pretrained=pretrained)
        self.decoder = MultiViewDecoder.from_assets(assets_dir, num_layers=num_layers)
        self.use_depth = use_depth
        self.img_size = img_size

    def forward(self, rgbd: torch.Tensor, proj: torch.Tensor, image_hw):
        B, N, C, H, W = rgbd.shape
        if not self.use_depth:
            rgbd = rgbd.clone()
            rgbd[:, :, 3] = 0.0                                  # RGB-only ablation
        x = rgbd
        if self.img_size is not None and self.img_size != H:
            x = F.interpolate(rgbd.reshape(B * N, C, H, W), size=self.img_size,
                              mode="bilinear", align_corners=False)
            x = x.reshape(B, N, C, self.img_size, self.img_size)
        feats = self.backbone(x)                                # (B,N,256,h,w)
        # image_hw = ORIGINAL size the proj matrices map into (not img_size)
        return self.decoder(feats, proj, image_hw)
