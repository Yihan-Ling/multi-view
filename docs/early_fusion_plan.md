# Early-Fusion Depth Model — Plan

Owner: Carson. Sibling track: **late fusion** (teammate). Both feed the same
multi-view head-pose pipeline; the deliverable of this track is the
**early-vs-late fusion comparison** under a shared eval protocol.

## 1. What "early fusion" means here

Depth enters as a **4th input channel**, fused with RGB *before* any learned
features:

$$x_{\text{in}} = \operatorname{concat}(\text{RGB}, \text{D}) \in \mathbb{R}^{H\times W\times 4}
\;\xrightarrow{\;\text{conv1 (4-ch)}\;}\; \text{ResNet-50} \;\rightarrow\; \dots$$

Contrast — **late fusion** keeps a 3-channel RGB backbone and injects depth
downstream, in the geometric stage (lifting 2D features/landmarks to 3D using
the depth map). The single controlled variable between the two tracks is
*where depth enters*; everything downstream (cross-view decoder, 3D landmarks,
pose fit) is shared.

## 2. Current status (implemented)

The single-view early-fusion backbone already exists in `multi_view/`:

- [`backbone.py`](../multi_view/backbone.py) — `RGBDPoseResNet50`: 4-channel
  `conv1` stem $\rightarrow$ ResNet-50 stages 1–4 $\rightarrow$ PoseResNet-style
  deconv head (Xiao et al., ECCV 2018). Output stride $/4$.
- [`weight_init.py`](../multi_view/weight_init.py) —
  `init_conv1_4ch_from_pretrained`: copies the RGB pretrained `conv1` weights
  into channels 0–2 and replicates the **red** channel into channel 3 (depth).
  Temporary scaffold; long-term intent is train-from-scratch.
- [`model.py`](../multi_view/model.py) — `SingleViewLandmarkModel`:
  backbone $\rightarrow$ MLP head $\rightarrow$ $(B, 68, 3)$.
- [`scripts/test_model_1/`](../scripts/test_model_1/) — unit tests:
  forward-shape, backward (gradient flow), overfit-a-single-sample.

Gap: the backbone has only been exercised on **random tensors**. It has not yet
seen a real FaceScape RGBD sample, and the multi-view aggregation / 3D / pose
stages are not built.

## 3. Plan to proceed

Phased, one variable at a time. Phases 1–2 are single-view (de-risk the fusion
mechanics); 3–4 add the multi-view geometry; 5 is the actual experiment.

1. **Real-data single-view smoke.** Feed one real FaceScape RGBD view
   (`rgb.png` + rendered depth, via [`multi_view/data/facescape.py`](../multi_view/data/facescape.py))
   through `SingleViewLandmarkModel`. Confirm shapes, depth normalization, and
   that the overfit test still passes on a real sample (not random noise).
2. **Single-view training on synthetic.** Train the early-fusion model on the
   FaceScape virtual RGBD renders (subject-disjoint split). Report 3D-landmark
   error (mm). This validates that depth-as-4th-channel actually *helps* vs an
   RGB-only ablation of the same backbone.
3. **Multi-view aggregation.** Add the transformer decoder that fuses per-view
   backbone features across $V$ views into a single 3D-landmark prediction in
   the world frame (per the MvP/MVGFormer study). Backbone weights shared
   across views.
4. **Pose readout.** Fit the head model to the predicted 3D landmarks to recover
   6-DoF pose; report rotation (deg) and translation (mm) error.
5. **Early-vs-late comparison.** Run this track and the teammate's late-fusion
   track through the **same** data split, decoder, and metrics. Deliverable =
   the head-to-head table (3D-landmark mm, pose deg/mm, and cost:
   params / FLOPs / latency).

## 4. Evaluation protocol (shared with late-fusion track)

- **Split:** subject-disjoint train/val on FaceScape (no identity leakage).
- **Metrics:** 3D-landmark error (mm); 6-DoF pose error (rotation deg,
  translation mm); efficiency (params, FLOPs, latency).
- **Ablation:** early-fusion RGBD vs RGB-only backbone (isolates the depth
  channel's contribution) — mirrors the eval late fusion is measured on.

## 5. Open decisions

- **4th-channel init:** train-from-scratch vs the current red-channel copy.
  Long-term intent is from-scratch; keep the copy only as a warm-start crutch
  while iterating.
- **Depth encoding:** raw metric depth vs normalized/inverse depth vs
  log-depth as the 4th channel — affects conv1 statistics and cross-subject
  scale invariance.
- **Depth holes:** FaceScape TU mesh has no eyeballs $\rightarrow$ depth$=0$
  holes in the eye region (same artifact that bit the HRNet landmark work).
  Decide a fill/mask policy for the depth channel before training.
