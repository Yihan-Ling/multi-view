import torch
from torch import nn
from torchvision.models import resnet50


class RGBDPoseResNet50(nn.Module):
    """ResNet-50 with a 4-channel (RGBD) first conv and a PoseResNet-style
    deconv head (Xiao, Wu & Wei, ECCV 2018, "Simple Baselines for Human Pose
    Estimation and Tracking"). Output spatial resolution is input / 4.

    From-scratch only: the 4-channel conv1 and the whole ResNet body are randomly
    initialized (no ImageNet weights). This is the committed training regime; the
    old ImageNet-pretrained 4-ch bridge was scaffolding and has been removed.
    """

    def __init__(
        self,
        num_deconv_layers: int = 3,
        deconv_channels: int = 256,
    ) -> None:
        super().__init__()
        rn = resnet50(weights=None)
        rn.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False) # 4-ch RGBD input, random init

        # Include Stage 1-5
        self.conv1 = rn.conv1
        self.bn1 = rn.bn1
        self.relu = rn.relu
        self.maxpool = rn.maxpool
        self.layer1 = rn.layer1
        self.layer2 = rn.layer2
        self.layer3 = rn.layer3
        self.layer4 = rn.layer4

        # Add 3 layers of  deconvolution
        self.deconv_head = self._build_deconv_head(
            in_channels=2048,
            num_layers=num_deconv_layers,
            out_channels=deconv_channels,
        )
        self.out_channels = deconv_channels

    @staticmethod
    def _build_deconv_head(in_channels: int, num_layers: int, out_channels: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        ch_in = in_channels
        for _ in range(num_layers):
            layers.append(
                nn.ConvTranspose2d(
                    ch_in, out_channels, kernel_size=4, stride=2, padding=1, bias=False
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            ch_in = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.deconv_head(x)
        return x


class MultiViewBackbone(nn.Module):
    """Phase 3 - run the shared RGBD backbone over all N views of each sample.

        input   rgbd   (B, N, 4, H, W)     B samples, N views each
        output  feats  (B, N, C, H/4, W/4)  C = backbone.out_channels (256)

    The backbone is view-agnostic: the SAME weights process every view. So
    instead of looping over views, we fold the N-view axis into the batch
    dimension, do one backbone pass, then unfold. The only thing to get right is
    the bookkeeping -- the unfold must restore (B, N, ...) in the same order the
    fold flattened them.
    """

    def __init__(self, deconv_channels: int = 256) -> None:
        super().__init__()
        self.backbone = RGBDPoseResNet50(deconv_channels=deconv_channels)
        self.out_channels = self.backbone.out_channels

    def forward(self, rgbd: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = rgbd.shape

        # BLANK 1: fold the N view axis into the batch dim so one backbone pass
        # handles all views. Target shape (B*N, C, H, W).
        # Hint: reshape/view is row-major -> B is the OUTER index, N the INNER, so
        # element (b, n) lands at row b*N + n. Keep that order for the unfold.
        x = rgbd.reshape(B*N, C, H, W)

        feats = self.backbone(x)

        # BLANK 2: unfold the batch dim back to (B, N, out_channels, H/4, W/4).
        # Use feats' own spatial size (H/4, W/4), not H, W.
        feats = feats.reshape(B, N, self.out_channels, feats.shape[-2], feats.shape[-1])

        return feats
