"""Phase 5 KEYSTONE gate - overfit ONE multi-view sample.

Wires backbone -> decoder -> losses and trains on a single sample. If the wiring
is correct, the last-layer 3D error should collapse toward ~0 and each of the 4
decoder layers should show progressively smaller error (refinement working).

If this can't overfit one sample, do NOT start a full training run - the wiring is
wrong. Uses the real backbone, so run it yourself (GPU strongly recommended):

    .venv/bin/python scripts/test_mv/test_overfit.py
"""

import _init_paths  # noqa: F401
import os

import torch

from _init_paths import REPO_ROOT
from multi_view.data.facescape_dataset import MultiViewFaceScape, discover_subjects
from multi_view.losses import decoder_losses, mpjpe_mm
from multi_view.mv_model import MultiViewLandmark3D

ROOT = REPO_ROOT / "data" / "facescape" / "virtual_camera_data"
ASSETS = REPO_ROOT / "multi_view" / "assets"
STEPS = 400


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    subs = discover_subjects(ROOT)[:1]
    s = MultiViewFaceScape(ROOT, subs)[0]
    rgbd = s["rgbd"].unsqueeze(0).to(device)
    proj = s["proj"].unsqueeze(0).to(device)
    gt_3d = s["landmarks_3d"].unsqueeze(0).to(device)
    gt_2d = s["landmarks_2d"].unsqueeze(0).to(device)
    vis = s["vis"].unsqueeze(0).to(device)
    image_hw = (rgbd.shape[-2], rgbd.shape[-1])

    model = MultiViewLandmark3D(ASSETS, num_layers=4, pretrained=False,
                                use_depth=True, img_size=256).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    init_err = None
    for step in range(1, STEPS + 1):
        opt.zero_grad()
        preds_3d, preds_2d = model(rgbd, proj, image_hw)
        losses = decoder_losses(preds_3d, preds_2d, gt_3d, gt_2d, vis)
        losses["total"].backward()
        opt.step()

        if step == 1:
            init_err = mpjpe_mm(preds_3d[-1], gt_3d)
        if step % 50 == 0 or step == 1:
            per_layer = [f"{mpjpe_mm(p, gt_3d):6.2f}" for p in preds_3d]
            print(f"step {step:4d}  loss {losses['total'].item():8.3f}  "
                  f"3D mm/layer [{' '.join(per_layer)}]")

    final = mpjpe_mm(preds_3d[-1], gt_3d)
    layer_errs = [mpjpe_mm(p, gt_3d) for p in preds_3d]
    print(f"\ninit last-layer 3D error: {init_err:.2f} mm  ->  final: {final:.2f} mm")
    print("final per-layer 3D error (should DECREASE across layers):",
          [round(e, 2) for e in layer_errs])
    if final < 5.0:
        print("KEYSTONE PASSED: model overfit one sample (last-layer < 5 mm).")
    else:
        print("KEYSTONE NOT MET: last-layer error >= 5 mm - inspect wiring/lr/steps.")


if __name__ == "__main__":
    main()
