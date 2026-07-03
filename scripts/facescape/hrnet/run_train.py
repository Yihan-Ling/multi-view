#!/usr/bin/env python3
"""Train HRNet with the on-the-fly FaceScapeAug loader -- no edits to vendored code.

Mirrors eval_real/run_eval.py: puts the HRNet `lib` on sys.path, swaps the dataset
registry so get_dataset() returns FaceScapeAug (live photometric augmentation) while
cfg.DATASET.DATASET stays '300W' (CSV format + flip index unchanged), then runs the
stock tools/train.py main().

Run from the MAIN repo root (the cfg's data/output paths are relative to it):

    .venv/bin/python scripts/facescape/hrnet/run_train.py \
        --cfg scripts/facescape/hrnet/face_alignment_facescape_w18_scratch.yaml
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
HR = os.path.join(REPO, "third_party", "HRNet-Facial-Landmark-Detection")

# HRNet's `lib` package and our facescape_aug must both be importable.
sys.path.insert(0, HR)
sys.path.insert(0, HERE)

# 1) Patch the registry BEFORE train.py binds the name via `from lib.datasets
#    import get_dataset` -- so the bound name already points at our loader.
import lib.datasets as D                     # noqa: E402
from facescape_aug import FaceScapeAug        # noqa: E402

D.get_dataset = lambda cfg: FaceScapeAug

# 2) Load the vendored training script as a module. Its `if __name__ ==
#    "__main__"` guard does NOT fire (name is "hrnet_train"), so nothing runs yet.
_spec = importlib.util.spec_from_file_location(
    "hrnet_train", os.path.join(HR, "tools", "train.py"))
_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train)
_train.get_dataset = lambda cfg: FaceScapeAug  # belt-and-suspenders override

# 3) Thin the per-epoch checkpointing WITHOUT editing vendored code. Stock
#    tools/train.py calls utils.save_checkpoint() every epoch, which writes a
#    full checkpoint_{epoch}.pth each time (60 large files for a 60-ep run). We
#    replace utils.save_checkpoint (train.py looks it up as an attribute on the
#    module at call time, so patching the module attribute takes effect) with a
#    policy that keeps latest.pth + model_best.pth EVERY epoch (resume works, no
#    best is ever missed) but only keeps the numbered archive every SAVE_EVERY.
import os                                        # noqa: E402
import torch                                     # noqa: E402
from lib.utils import utils as _utils            # noqa: E402

SAVE_EVERY = 5  # keep checkpoint_{epoch}.pth only when (epoch+1) % SAVE_EVERY == 0


def _save_checkpoint_every_n(states, predictions, is_best, output_dir,
                             filename="checkpoint.pth"):
    # current predictions: rolling, every epoch (unchanged from upstream)
    torch.save(predictions.cpu().data.numpy(),
               os.path.join(output_dir, "current_pred.pth"))

    # latest.pth as a REAL file (overwritten every epoch), so RESUME works even on
    # epochs where we skip the numbered archive. Upstream made it a symlink to the
    # numbered file, which would dangle on skipped epochs -- a real file is safe.
    latest_path = os.path.join(output_dir, "latest.pth")
    if os.path.islink(latest_path) or os.path.exists(latest_path):
        os.remove(latest_path)
    torch.save(states, latest_path)

    # best model: every epoch it improves (unchanged from upstream)
    if is_best and "state_dict" in states:
        torch.save(states["state_dict"].module,
                   os.path.join(output_dir, "model_best.pth"))

    # numbered archive: only every SAVE_EVERY completed epochs. states["epoch"] is
    # epoch+1, so this saves at epoch indices 4,9,...,59 for a 60-epoch run (and the
    # final epoch, since 60 % 5 == 0). final_state.pth is still saved after the loop.
    if states.get("epoch", 0) % SAVE_EVERY == 0:
        torch.save(states, os.path.join(output_dir, filename))


_utils.save_checkpoint = _save_checkpoint_every_n

if __name__ == "__main__":
    _train.main()  # consumes --cfg from sys.argv
