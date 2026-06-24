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

if __name__ == "__main__":
    _train.main()  # consumes --cfg from sys.argv
