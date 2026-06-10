import scripts.test_model_1._init_paths as _init_paths  # noqa: F401

import math

import torch
import torch.nn.functional as F

from multi_view import SingleViewLandmarkModel


def main() -> None:
    # torch.manual_seed(0)
    batch, num_landmarks = 2, 68

    model = SingleViewLandmarkModel(num_landmarks=num_landmarks, pretrained_backbone=True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(batch, 4, 256, 256)
    target = torch.randn(batch, num_landmarks, 3)

    pred = model(x) # Step 1: Forward pass
    loss_before = F.mse_loss(pred, target)  # Step 2: Compute loss
    assert math.isfinite(loss_before.item()), f"loss not finite: {loss_before.item()}"

    optimizer.zero_grad()
    loss_before.backward()  # Step 3: Backward pass

    # Picked 4 places to check the gradient
    grad_checks = {
        "backbone.conv1.weight": model.backbone.conv1.weight.grad,  # very first layer
        "backbone.layer4[-1].conv3.weight": model.backbone.layer4[-1].conv3.weight.grad,    # last residual chunk
        "backbone.deconv_head[0].weight": model.backbone.deconv_head[0].weight.grad,    # first deconv
        "head.mlp[-1].weight": model.head.mlp[-1].weight.grad,  # very last layer
    }
    for name, grad in grad_checks.items():
        assert grad is not None, f"{name}: grad is None"
        norm = grad.norm().item()
        assert norm > 0.0, f"{name}: grad norm is zero"
        print(f"  grad {name:40s} norm={norm:.4e}")

    optimizer.step()

    with torch.no_grad():
        loss_after = F.mse_loss(model(x), target)

    print(f"loss before step: {loss_before.item():.6f}")
    print(f"loss after step:  {loss_after.item():.6f}")
    print("test_02_backward: PASS")


if __name__ == "__main__":
    main()
