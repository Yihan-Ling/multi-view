import torch
from torch import nn


class MLPLandmarkHead(nn.Module):
    """Global-avg-pool + 2-layer MLP that regresses K x 3 landmark coordinates
    from a feature map. Baseline head; replaced by a transformer decoder once
    that test program lands.
    """

    def __init__(self, in_channels: int = 256, num_landmarks: int = 68, hidden: int = 256) -> None:
        super().__init__()
        self.num_landmarks = num_landmarks
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_landmarks * 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.pool(x).flatten(1)
        x = self.mlp(x)
        return x.view(b, self.num_landmarks, 3)
