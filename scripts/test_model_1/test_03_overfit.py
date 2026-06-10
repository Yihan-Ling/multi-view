import scripts.test_model_1._init_paths as _init_paths  # noqa: F401

import sys

import torch
import torch.nn.functional as F

from multi_view import SingleViewLandmarkModel


def main() -> None:
    # torch.manual_seed(0)
    num_landmarks = 68
    num_iters = 200
    log_every = 20
    target_loss = 1e-6

    model = SingleViewLandmarkModel(num_landmarks=num_landmarks, pretrained_backbone=True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(1, 4, 256, 256)
    target = torch.randn(1, num_landmarks, 3)
    
    last_loss = float("inf")
    
    # 200 iterations of forward - loss - backward -opitimize loop
    for it in range(1, num_iters + 1):
        pred = model(x)
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
        if it == 1 or it % log_every == 0:
            print(f"iter {it:4d}  loss={last_loss:.6e}")

    # Assert if overfitting on one set of data can converge the loss to zero
    print(f"final loss: {last_loss:.6e}  (target < {target_loss:.0e})")
    if last_loss < target_loss:
        print("test_03_overfit: PASS")
    else:
        print("test_03_overfit: FAIL (model did not overfit single sample)")
        sys.exit(1)


if __name__ == "__main__":
    main()
