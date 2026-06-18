# Desktop training brief (for a Claude Code session on the RTX 6000)

You are picking up an in-progress task on a desktop with an RTX 6000. The code was
prepared on a laptop and pushed via git; the dataset arrives separately on USB. This
file is your full context -- the laptop's session notes do not travel with the repo.

Read this top to bottom before doing anything. For setup *mechanics* (clone, patch,
dataset restore) follow [README.md](README.md); this file covers the *why*, how to
read the results, and how to decide what to run next.

## 1. The task and why it exists

We are running a **sim-to-real validation**:

- **Train** HRNetV2-W18 to regress **68 2D facial landmarks** on **synthetic** FaceScape
  renders (RGB only).
- **Test** the trained model on **real** face datasets (AFLW2000-3D, WFLW).
- **Metric:** inter-ocular **NME** (normalized mean error) = mean per-point pixel error
  divided by the inter-ocular distance $\lVert \text{lm}_{36} - \text{lm}_{45} \rVert$
  (outer eye corners). Lower is better; it is reported as a fraction (e.g. $0.03 = 3\%$).
- **The finding** we are after: the gap between our synthetic-trained NME and the
  published real-trained NME. That gap quantifies how far synthetic-only training gets
  us. The job of *this* desktop is the **train** half on real GPU hardware.

Your job here: run the full training, read the validation curve, and decide whether the
training recipe (epochs, LR schedule) needs adjusting -- then iterate.

## 2. What is already done (do not redo)

- The **data adapter** (`scripts/facescape/build_hrnet_landmark_dataset.py`) converts
  rendered FaceScape views into HRNet's 300W CSV format. Already run on the laptop; its
  output (`data/facescape/HRNet_train/{train.csv,val.csv,images/}`) arrives on USB.
  ~146 subjects, **2925 train images / 725 val images**, split subject-disjoint.
- A **smoke test passed on the laptop**: 1 epoch, data loads, loss drops, checkpoint
  saves. So the code path is known-good; you are not debugging plumbing, you are training.
- The training **config** and a required **crop patch** are in this folder (see below).

## 3. Environment setup

Follow [README.md](README.md). In short:

1. `bash scripts/facescape/hrnet/setup_hrnet.sh` -- clones HRNet into `third_party/`
   (gitignored), pins commit `f776dbe`, applies the crop fix, places the config.
2. Restore the dataset from USB into `data/facescape/HRNet_train/`.
3. Install deps: `torch torchvision opencv-python numpy pandas Pillow tensorboardX yacs hdf5storage`
   (match the torch build to the desktop CUDA).
4. **Recommended:** put the ImageNet W18 weights at
   `third_party/HRNet-Facial-Landmark-Detection/hrnetv2_pretrained/hrnetv2_w18_imagenet_pretrained.pth`.
   To train from scratch instead, set `MODEL.PRETRAINED: ''` in the config.

**Verify before training:** `data/facescape/HRNet_train/train.csv` and `val.csv` exist,
`images/` is populated, and `python -c "import torch; print(torch.cuda.is_available())"`
prints `True`.

## 4. Run training

From the repo root:

```bash
python third_party/HRNet-Facial-Landmark-Detection/tools/train.py \
  --cfg scripts/facescape/hrnet/face_alignment_facescape_w18.yaml
```

The config (`face_alignment_facescape_w18.yaml`) defaults: `BATCH_SIZE_PER_GPU: 16`,
`END_EPOCH: 60`, `LR: 0.0001`, `LR_STEP: [30, 50]`, `RESUME: false`. On the laptop's
8 GB GPU one epoch was ~1 min; the RTX 6000 has far more memory and throughput, so
expect well under that per epoch and a higher feasible batch size.

Notes:
- `RESUME: false` is intentional -- upstream `save_checkpoint` writes a broken
  `latest.pth` symlink, so resume is unreliable. Each run starts fresh.
- Outputs land in `output/300W/face_alignment_facescape_w18/` and
  `log/300W/hrnet/face_alignment_facescape_w18_<timestamp>/`. Both are gitignored. The
  `300W` in the path is just because the config reuses the 300W data loader.

## 5. Analyze the output

Per-epoch validation NME is stored two ways (no CSV is emitted):

**Text log (fastest):**
```bash
grep "Test Epoch" output/300W/face_alignment_facescape_w18/*_train.log
```
Each line looks like:
```
Test Epoch 12 time:... loss:... nme:0.0421 [008]:0.0123 [010]:0.0061
```
- `nme` = inter-ocular NME for that epoch (the number you track).
- `[008]` / `[010]` = failure rates (fraction of points with NME > 0.08 / > 0.10).

**TensorBoard (the plottable curve):**
```bash
tensorboard --logdir log/
```
Scalars: `valid_nme`, `valid_loss`, `train_loss`.

**Best checkpoint:** `output/300W/face_alignment_facescape_w18/model_best.pth` is saved
whenever val NME improves -- this is the model you keep, not `final_state.pth`.

Rough reading of the numbers: synthetic-to-synthetic val NME will look optimistic (train
and val are both FaceScape). The honest accuracy number comes later from the real test
sets, not from this val curve. Here the val curve is for **convergence diagnosis only**.

## 6. Use the output to decide next steps

Plot/scan val NME vs epoch and classify the curve:

| What the val NME curve does | Diagnosis | Action |
|---|---|---|
| Still clearly dropping at epoch 60 | Undertrained | Raise `END_EPOCH` (e.g. 90) and **move `LR_STEP` accordingly**, e.g. `[45, 75]`. Re-run. |
| Flattens well before 60 and stays flat | Converged early | Fine as-is; optionally trim `END_EPOCH` to save time on future runs. |
| Drops, then **rises** late | Overfitting | The best epoch was earlier -- `model_best.pth` already holds it. Consider more augmentation or fewer epochs; do not just train longer. |
| Jumps down sharply right after epoch 30 / 50 | Normal -- that is the LR drop kicking in | No action; confirms the schedule is doing its job. |

Key coupling to respect: **`END_EPOCH` and `LR_STEP` are not independent.** The LR drops
$10^{-4} \to 10^{-5}$ at epoch 30 and $\to 10^{-6}$ at 50; `END_EPOCH` should sit a bit
past the last drop so the model settles at the lowest LR. If you extend training, scale
the steps too (keep the last drop ~10 epochs before the end).

If a run OOMs (unlikely on the RTX 6000), lower `BATCH_SIZE_PER_GPU`; throughput is
GPU-bound so total time barely changes. If you raise batch size a lot, consider nudging
`LR` up proportionally.

**Report back to the user** after a run with: the best val NME and which epoch it hit,
whether the curve had converged, and your recommended `END_EPOCH`/`LR_STEP` for the next
run (if any). Do not silently launch many long runs -- propose the change first.

## 7. Guardrails (things that will silently break results)

- **Do not replace the crop fix with the repo's `crop_v2`.** The patch
  (`fix_transforms.py`) rewrites `crop()` to `cv2.warpAffine` using the **same
  `get_transform` matrix the landmark targets use**. `crop_v2` uses a different affine
  and would misalign images vs labels -- training would "work" but learn garbage.
- **Do not commit `output/` or `log/`** (already gitignored).
- **Known data caveat:** ~76% of views have the chin tip clipped by the renderer; those
  off-image points are marked invalid (sentinel, skipped in the loss). Expect weaker
  lower-jaw accuracy -- this is a known limitation, not a bug to chase.
- **Do not change the adapter or re-render** here. The dataset is fixed input for this
  step; data changes happen on the laptop.

## 8. After a good training run

The next milestone (likely back on the laptop or wherever the real datasets live) is the
**real test step**: evaluate `model_best.pth` on AFLW2000-3D and WFLW with the same
inter-ocular NME. To support that, make sure `model_best.pth` (and the `log/` folder if
the user wants the curve) is exported off this desktop. That converter does not exist
yet -- it is the next thing to build, not part of this desktop's job.
