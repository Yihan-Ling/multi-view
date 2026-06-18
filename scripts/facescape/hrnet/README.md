# HRNet training on the FaceScape synthetic set (desktop runbook)

Train HRNetV2-W18 to regress 68 2D landmarks on the synthetic FaceScape renders,
as the train side of the sim-to-real validation (test later on AFLW2000-3D / WFLW).

## What moves how

| Thing | Route | Why |
|---|---|---|
| This repo's code (adapter, config, setup, fix) | **git clone** | tracked in git |
| HRNet model repo | **`setup_hrnet.sh`** clones it | `third_party/` is gitignored |
| Dataset (`HRNet_train/`, ~700 MB) | **USB** | `/data/` is gitignored; also gated FaceScape data |
| ImageNet W18 weights (`.pth`) | **manual download** | `*.pth` is gitignored |

## Build the dataset (on the laptop, before the trip)

```bash
python -c "import sys; sys.path.insert(0,'scripts/facescape'); \
  import build_hrnet_landmark_dataset as b; \
  b.main('data/facescape/virtual_camera_data','data/facescape/HRNet_train')"
tar czf HRNet_train.tar.gz -C data/facescape HRNet_train      # put this on the USB
```

## On the desktop

```bash
# 1. clone your repo
git clone git@github.com:Yihan-Ling/multi-view.git
cd multi-view

# 2. clone + patch HRNet, place the config
bash scripts/facescape/hrnet/setup_hrnet.sh

# 3. restore the dataset from USB
mkdir -p data/facescape
tar xzf /path/to/usb/HRNet_train.tar.gz -C data/facescape
#   -> data/facescape/HRNet_train/{train.csv,val.csv,images/}

# 4. (recommended) ImageNet-pretrained W18 backbone
#    From https://github.com/HRNet/HRNet-Image-Classification (HRNetV2-W18),
#    save as:
#    third_party/HRNet-Facial-Landmark-Detection/hrnetv2_pretrained/hrnetv2_w18_imagenet_pretrained.pth
#    To skip and train from scratch instead, set MODEL.PRETRAINED: '' in the config.

# 5. train (from repo root)
python third_party/HRNet-Facial-Landmark-Detection/tools/train.py \
  --cfg scripts/facescape/hrnet/face_alignment_facescape_w18.yaml
```

## Python deps

```bash
pip install torch torchvision opencv-python numpy pandas Pillow \
            tensorboardX yacs hdf5storage
```
(Match the torch build to the desktop's CUDA.)

## Smoke test first

Before the full 60-epoch run, confirm data loads and loss drops: temporarily set
`TRAIN.END_EPOCH: 1` in the config (or Ctrl-C after ~50 iterations) and watch the
printed loss decrease. Outputs land in `output/` and `log/` (both gitignored).

## Notes

- `setup_hrnet.sh` and `fix_transforms.py` are idempotent -- safe to re-run.
- `TRAIN.RESUME` is `false`: each run starts fresh. An interrupted run restarts from
  epoch 0 (upstream `save_checkpoint` writes a broken `latest.pth` symlink, so resume
  is unreliable; starting fresh sidesteps it).
- The crop fix is required: upstream `crop()` uses `scipy.misc` (gone in scipy>=1.12)
  and `np.math` (gone in numpy>=2.0). The patch rewrites it with cv2 using the same
  `get_transform` matrix the landmark targets use, so image and labels stay aligned.
- Known data caveat: ~76% of views have the chin tip clipped by the renderer; those
  points are marked invalid (skipped in the loss), so expect weaker lower-jaw
  accuracy. See the dataset adapter and project notes.
```
