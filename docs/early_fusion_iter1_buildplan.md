# Iteration 1 — Build Plan (early-fusion, MVGFormer imitation)

Target pipeline:

> RGBD (4-ch, from-scratch) $\rightarrow$ ResNet-50 + deconv feature map
> $\rightarrow$ PyTorch projective-attention decoder (mean-face query init)
> $\rightarrow$ batch-DLT triangulation $\rightarrow$ **3D landmarks**, trained on FaceScape.

Scope: **stops at 3D landmarks.** No 6-DoF, no bone loss, no custom CUDA.

**Guiding principle — build the geometry first, verify each piece in isolation
before wiring.** A 1-pixel projection bug becomes a degree-scale 3D error after
triangulation and is nearly impossible to debug once the network is training.
So Phases 0–2 have *no learning at all* — they nail the camera math and data.

Each phase has: **Goal / Build / Verify (gate)**. Do not start a phase until the
previous phase's gate passes.

---

## Phase 0 — Camera geometry foundation (no learning)

**Goal.** Two functions that are inverses of each other: project a 3D world
point into each camera's pixel, and triangulate pixels back to 3D.

**Build.**
- `project(points_3d, P)` — world 3D $\rightarrow$ 2D pixels, where the
  projection matrix is $P = K[R\mid t] \in \mathbb{R}^{3\times4}$. Homogeneous
  multiply, then divide by the 3rd coordinate.
- `triangulate_dlt(points_2d, P_list)` — the batch DLT: stack the
  $2N\times4$ system $A\mathbf{X}=0$ (one pair of rows per view), solve by
  `torch.linalg.svd`, take the last right-singular vector, de-homogenize.
  Support optional per-view confidence weights (scale each view's rows).

**Verify (gate) — the round-trip test.** This is the single most important test
in the whole iteration.
1. Make up a synthetic 3D point (or the 68 mean-face points).
2. Project it into all $N$ real FaceScape cameras $\rightarrow$ 2D pixels.
3. Triangulate those pixels back $\rightarrow$ 3D.
4. Assert recovered 3D $\approx$ original (error $< 10^{-4}$).

If this fails, everything downstream is wrong — stop and fix conventions
(world-vs-camera, the CV convention from `multi_view/data/facescape.py`, pixel
origin). Keep this test and run it continuously.

---

## Phase 1 — Multi-view RGBD data pipeline (no learning)

**Goal.** A `Dataset` that yields one *multi-view sample*: the same face seen
from $N$ views, with everything the model and losses need.

**Build.** `__getitem__` returns a dict per sample:
- `rgbd`: $(N, 4, H, W)$ — per view, RGB (`rgb.png`) + depth stacked as the 4th
  channel. Reuse `discover_views` / the FaceScape reader.
- `proj`: $(N, 3, 4)$ — projection matrix per view (from `K`, `Rt`).
- `landmarks_3d`: $(68, 3)$ — GT in the **world** frame.
- `landmarks_2d`: $(N, 68, 2)$ — GT per view (project the 3D GT, or read
  `landmarks_2d.npy`).
- `vis`: $(N, 68)$ — per-view visibility (1 if the landmark projects inside the
  image and is not self-occluded; start with the in-image test only).

Two sub-decisions to make here:
- **Depth normalization.** RGB is $\sim[0,1]$; raw depth in millimetres is
  hundreds and would dominate `conv1`. Default: subtract the median face depth
  and divide by a fixed scale (~$200$ mm) so the channel is $\sim O(1)$.
  Record the exact transform; it must match at train and test.
- **Depth holes.** TU mesh has no eyeballs $\rightarrow$ `depth == 0` holes in
  the eye region (same artifact as the HRNet work). Default: fill interior
  holes (`binary_fill_holes`) and set them to the face's median depth, not 0.

Use the existing **subject-disjoint** split logic.

**Verify (gate).** Visual overlay: project `landmarks_3d` into each view with the
Phase-0 `project()` and draw the dots on `rgb.png`. Eyeball that all 68 land on
the correct facial features in every view. Also print min/max of the normalized
depth channel (should be $\sim O(1)$, no NaNs).

---

## Phase 2 — Mean-face template (no learning)

**Goal.** The average face the decoder starts every query from.

**Build.** Average `landmarks_3d` across all **training** subjects (align/center
first so you average shapes, not positions). Save as an artifact
(`mean_face_68.npy`). Also define the 3D **query space**: a box
(`SPACE_SIZE` $\approx 300$ mm cube) centered on the face
(`SPACE_CENTER` = mean-face centroid). Reference points will be expressed
normalized to $[0,1]^3$ inside this box.

**Verify (gate).** 3D scatter plot of the mean face — eyeball that it looks like
a face (eyes/nose/mouth in the right place), not a collapsed blob.

---

## Phase 3 — Backbone feature map (mostly done)

**Goal.** Confirm the backbone emits the feature map the decoder samples.

**Build.** `RGBDPoseResNet50.forward` already returns the
$256\times64\times64$ feature map (the deconv output) — good. Just make sure the
multi-view path feeds it $(N,4,H,W)$ and gets $(N,256,64,64)$ back (fold $N$ into
the batch dim). Keep the `MLPLandmarkHead` untouched as the separate single-view
baseline. Drop `init_conv1_4ch_from_pretrained` on the from-scratch path
(random-init all 4 channels).

**Verify (gate).** Feed one Phase-1 sample; assert output shape
$(N, 256, 64, 64)$, finite values.

---

## Phase 4 — Projective attention module (PyTorch)

**Goal.** Given 3D query points, gather image features from every view at the
places those queries project to.

**Build.** `ProjAttn(query_ref_3d, feat_maps, proj)`:
1. Project each query's 3D reference point into each view (Phase-0 `project`),
   normalize to $[-1,1]$ for `F.grid_sample`.
2. `F.grid_sample` the feature map at that 2D location per view $\rightarrow$
   one feature vector per (query, view).
3. (Optional, faithful to MVGFormer) concat the camera **ray direction** at that
   pixel before a linear layer — geometric position encoding.
4. Aggregate the per-view features into one feature per query (attention-weighted
   sum over views). Start simple: sample the single projected point; add small
   learnable sampling offsets later if needed.

This is the pure-PyTorch replacement for the CUDA deformable kernel.

**Verify (gate).** Put a query exactly at a known GT landmark's 3D position; the
sampled 2D location should match that landmark's GT 2D pixel (reuses Phase-0).
Shapes: $(\text{queries}, d_{\text{model}})$.

---

## Phase 5 — Decoder layer + stack (the core)

**Goal.** One refinement iteration, stacked $L=4$ times, each improving the 3D
estimate.

**Build.** `DecoderLayer.forward(query_feat, ref_3d, feat_maps, proj)` does:
1. `ProjAttn` (Phase 4) $\rightarrow$ per-query features.
2. **Update features**: self-attention across the 68 queries + FFN (they share
   information — e.g. eye corners constrain each other).
3. **Predict per-view 2D offsets + confidences**: small heads that nudge each
   query's projected 2D point and score how trustworthy each view is.
4. **Triangulate** (Phase-0 DLT, confidence-weighted) $\rightarrow$
   `new_ref_3d`.
5. Return updated features + `new_ref_3d`; feed into the next layer.

Init: queries = learned embeddings; `ref_3d` = mean-face template (Phase 2).

**Verify (gate).** Forward shapes; gradients flow (backward test);
**overfit one multi-view sample** — the 3D landmark output should converge to
that sample's GT (mirrors your existing `test_03_overfit`). If it can't overfit
one sample, the wiring is wrong.

---

## Phase 6 — Losses

**Goal.** Supervise 3D and the per-view 2D that drives triangulation.

**Build.**
- 3D landmark L1: `L1(pred_3d, gt_3d)`.
- Per-view 2D-offset L1: `L1(refined_2d, gt_2d)` masked by `vis` — critical,
  because the 2D offsets are what triangulation consumes.
- Sum both across **all $L$ decoder layers** (auxiliary/deep supervision;
  standard DETR trick to stabilize training).

**Verify (gate).** On the overfit sample, total loss $\rightarrow \approx 0$ and
each layer's 3D error shrinks across the 4 layers (refinement is working).

---

## Phase 7 — Train, evaluate, and the ablation (the actual result)

**Goal.** A trained iteration-1 model and the number that matters.

**Build / run.**
- Train on FaceScape, subject-disjoint split, from scratch.
- Metric: mean 3D landmark error in **mm** on held-out subjects (MPJPE).
- **The experiment — RGBD vs RGB-only ablation.** Same everything, input = 4-ch
  vs 3-ch (zero the depth channel). The delta is early fusion's contribution and
  the headline of your track.

**Verify (gate).** Report held-out 3D mm error and the RGBD$-$RGB delta. Sanity:
error should be well below the $\sim300$ mm space size; a working model is
single-digit-to-low-tens of mm on clean synthetic data.

---

## Suggested order of attack (dependencies)

Phase 0 $\rightarrow$ 1 $\rightarrow$ 2 can proceed nearly independently after 0.
Phase 3 is quick. Phases 4 $\rightarrow$ 5 $\rightarrow$ 6 are the real modeling
work and are strictly sequential. Phase 7 needs everything.

The two keystones: **Phase 0's round-trip test** (protects all geometry) and
**Phase 5's overfit-one-sample test** (protects all wiring). If both pass, the
rest is tuning.
