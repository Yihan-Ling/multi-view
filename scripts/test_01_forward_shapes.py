import _init_paths  # noqa: F401

import torch

from multi_view import SingleViewLandmarkModel


def main() -> None:
    batch, channels, height, width = 2, 4, 256, 256
    num_landmarks = 68

    model = SingleViewLandmarkModel(num_landmarks=num_landmarks, pretrained_backbone=True)
    model.eval()

    x = torch.randn(batch, channels, height, width)

    with torch.no_grad():
        c1 = model.backbone.conv1(x)
        assert c1.shape == (batch, 64, 128, 128), f"conv1 out {tuple(c1.shape)}"

        feat = model.backbone(x)
        assert feat.shape == (batch, 256, 64, 64), f"backbone out {tuple(feat.shape)}"

        pred = model(x)
        assert pred.shape == (batch, num_landmarks, 3), f"model out {tuple(pred.shape)}"

    print(f"conv1 output:     {tuple(c1.shape)}")
    print(f"backbone output:  {tuple(feat.shape)}")
    print(f"model output:     {tuple(pred.shape)}")
    print("test_01_forward_shapes: PASS")


if __name__ == "__main__":
    main()
