import torch
from torch import nn

from multi_view.backbone import RGBDPoseResNet50
from multi_view.head import MLPLandmarkHead


class SingleViewLandmarkModel(nn.Module):
    """Day-1 baseline: one RGBD view -> backbone -> MLP head -> (B, K, 3)."""

    def __init__(
        self,
        num_landmarks: int = 68,
        pretrained_backbone: bool = True,
        deconv_channels: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = RGBDPoseResNet50(
            deconv_channels=deconv_channels, pretrained=pretrained_backbone
        )
        self.head = MLPLandmarkHead(
            in_channels=self.backbone.out_channels, num_landmarks=num_landmarks
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
