import torch
from torch import nn


def init_conv1_4ch_from_pretrained(
    new_conv: nn.Conv2d, pretrained_weight: torch.Tensor
) -> None:
    """Copy RGB pretrained conv1 weights into a 4-channel conv, replicating
    the red channel into the new depth channel.

    Temporary choice for short-term iteration; the long-term plan is to train
    from scratch on a real RGB-D dataset (see [[from-scratch-intent]] memory).
    """
    # Sanity Checks
    if new_conv.in_channels != 4:
        raise ValueError(f"expected new_conv.in_channels == 4, got {new_conv.in_channels}")
    if pretrained_weight.shape[1] != 3:
        raise ValueError(
            f"expected pretrained_weight shape (out, 3, k, k), got {tuple(pretrained_weight.shape)}"
        )
    # copy RGB channels as is and R channel into depth    
    with torch.no_grad():
        new_conv.weight[:, :3].copy_(pretrained_weight)
        new_conv.weight[:, 3:4].copy_(pretrained_weight[:, 0:1])
