#!/usr/bin/env python3
"""Thin eval driver: reuse HRNet's lib (model build + Face300W loader + NME
inference) but load OUR checkpoint formats, which stock tools/test.py can't on
torch>=2.6 (weights_only default flipped; our model_best.pth is a pickled
HighResolutionNet object, not a plain state_dict).

This adds NO new eval logic -- compute_nme/decode_preds/inference all come from
the HRNet repo. It only fixes checkpoint loading + prints the NME line.

Run from repo root:
  PYTHONPATH=third_party/HRNet-Facial-Landmark-Detection \
  .venv/bin/python scripts/facescape/hrnet/eval_real/run_eval.py \
    --cfg  scripts/facescape/hrnet/eval_real/face_alignment_aflw2000_w18.yaml \
    --model-file output/hrnet/forte_trained/300W/face_alignment_facescape_w18/model_best.pth
"""
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import lib.models as models
from lib.config import config, update_config
from lib.datasets import get_dataset
from lib.core import function


def load_state_dict_any(path):
    """Return a plain (no 'module.' prefix) state_dict from any of our .pth
    variants: pickled HighResolutionNet / DataParallel object, {'state_dict':..}
    checkpoint dict, or a bare state_dict."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, nn.DataParallel):
        return obj.module.state_dict()
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    # strip a possible 'module.' prefix from a raw state_dict
    return {k.replace("module.", "", 1): v for k, v in obj.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--model-file", required=True)
    args = ap.parse_args()
    # update_config reads args.cfg (mirrors how tools/test.py calls it)
    update_config(config, args)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()

    model = models.get_face_alignment_net(config)
    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()
    model.module.load_state_dict(load_state_dict_any(args.model_file))

    dataset_type = get_dataset(config)
    test_loader = DataLoader(
        dataset=dataset_type(config, is_train=False),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False, num_workers=config.WORKERS, pin_memory=config.PIN_MEMORY,
    )

    nme, _ = function.inference(config, test_loader, model)
    print(f"\nFINAL  nme={nme:.4f}  model={args.model_file}")


if __name__ == "__main__":
    main()
