# Design decisions and rejected alternatives

Living log of the architectural forks we have already chosen, the alternatives that were on the table at the time, and what would trigger us to revisit each one. Add a new section every time we pick between non-trivial alternatives so we don't re-litigate.

Companion doc: [docs/references/mvp_mvgformer_study.md](../references/mvp_mvgformer_study.md). The v0 pipeline this serves is recorded as the `pipeline-v0` project memory.

---

## 1. RGBD fusion strategy — 4-channel input

**Chosen:** Concatenate depth as a 4th channel and feed `(R, G, B, D)` into a modified ResNet-50 first conv. Implementation: [multi_view/backbone.py](../../multi_view/backbone.py).

**Alternatives considered:**

| Option | What it commits to | Why we didn't pick it |
|---|---|---|
| Two-stream RGB ResNet + Depth ResNet, late fusion | Separate backbones per modality, fuse feature maps before the decoder | ~2x backbone compute, extra fusion module to design; preserves RGB pretraining fully |
| RGB-only backbone, depth as geometry signal | Backbone never sees depth; depth becomes a 3D prior (unprojected point cloud) or an auxiliary loss | Backbone can't learn depth-conditioned features; harder to integrate cleanly in single-view setting |
| RGB + surface-normals (6-channel) | Convert depth to surface normals via local gradients, concat with RGB | Adds a normals-estimation step with its own noise; same pretraining-loss problem as 4-channel |

**Revisit if:** the modified-conv1 init proves to be a meaningful pretraining-loss handicap when we measure on a real dataset, or when we're forced to switch to a depth-aware geometry-first formulation (e.g., to handle very wide camera baselines).

---

## 2. First-test scope — single-view

**Chosen:** Single RGBD view through backbone + head, no camera projection, no multi-view fusion. Implementation: [multi_view/model.py](../../multi_view/model.py).

**Alternatives considered:**

| Option | What it commits to | Why we didn't pick it |
|---|---|---|
| Multi-view from the start | N views with synthetic camera params, shared-weight backbones, multi-view fusion in the decoder | Too many moving parts to debug at once for the first program |
| Two scripts side by side | Both single-view and multi-view scripts at once, sharing backbone code | Slight overhead; deferred until single-view is stable |

**Revisit:** This is "deferred", not "rejected forever" — multi-view is the eventual target. Next milestone after the transformer-decoder test program.

---

## 3. Backbone trim — full ResNet-50 + PoseResNet-style deconv head

**Chosen:** Keep ResNet-50 through layer4, drop avgpool + FC, append 3 transposed-conv layers (each kernel=4, stride=2, padding=1, 256 output channels with BN + ReLU). For a $256 \times 256$ input this gives a $64 \times 64 \times 256$ feature map. Implementation: [multi_view/backbone.py](../../multi_view/backbone.py).

The deconv-head pattern is **Xiao, Wu & Wei, "Simple Baselines for Human Pose Estimation and Tracking", ECCV 2018** ([arXiv:1804.06208](https://arxiv.org/abs/1804.06208)). Both MvP and MVGFormer port this verbatim as their `pose_resnet.py` ([study doc §1.4, §2.4](../references/mvp_mvgformer_study.md)).

**Alternatives considered:**

| Option | Output shape (256×256 input) | Why we didn't pick it |
|---|---|---|
| layer4 only, no upsampling | $8 \times 8 \times 2048$ | Too coarse spatially — each token covers ~32 input pixels, bad for landmark localization |
| layer3 only | $16 \times 16 \times 1024$ | Better spatial resolution but loses high-level semantic abstraction; not a standard pattern |
| Multi-scale FPN (layer2 + layer3 + layer4) | three maps, ~5376 total tokens | More compute; better suited to a decoder that can attend across scales (which we don't have yet) |

**Revisit if:** the transformer decoder lands and we measure that landmark localization is bottlenecked by feature resolution; or if we adopt MVGFormer-style projective attention that benefits from a feature pyramid.

---

## 4. First-test "decoder" — plain MLP head

**Chosen:** Global avgpool the backbone feature map to $(B, 256)$, then a 2-layer MLP $\to (B, K \times 3) \to$ reshape to $(B, K, 3)$. Implementation: [multi_view/head.py](../../multi_view/head.py).

**Alternatives considered:**

| Option | What it commits to | Why we didn't pick it for the first test |
|---|---|---|
| DETR-style transformer decoder | $K$ learnable query tokens, self-attn + cross-attn to flattened backbone features | The right next step, but more complex than needed to validate the backbone wiring |
| MVGFormer-style projective-attention decoder | Queries carry 3D positions, project into views, sample features at projected pixels | Requires camera params and multi-view setup; v2+, after single-view + DETR-style work |

**Revisit:** Explicitly a baseline. The next test program replaces the MLP head with a DETR-style transformer decoder. The MLP version stays in the repo as a regression baseline.

---

## 5. Conv1 init — replicate red into channel 4 (temporary)

**Chosen:** Load `torchvision.models.ResNet50_Weights.IMAGENET1K_V2`, copy the original conv1 weights into channels 0–2, copy channel 0 (red) again into channel 3 (depth). Implementation: [multi_view/weight_init.py](../../multi_view/weight_init.py).

**Reasoning:** Depth maps have edge / gradient statistics broadly similar to a grayscale image. The red channel is a common choice in the RGBD-CNN literature; the mean across (R,G,B) is also defensible. We pick red for simplicity.

**Alternatives considered:**

| Option | Why we didn't pick it |
|---|---|
| Zero-init the 4th channel | Slower start; network has to learn from scratch that depth carries signal |
| Full random init (no pretraining) | Forces real data to be available before any meaningful training — too slow for current iteration phase |
| Load PoseResNet-50 panoptic weights (from MvP) | Closer task than ImageNet, but adds an external-download dependency and the weights are for body pose, not faces |

**Revisit when:** we have a real RGB-D face dataset. The user has explicitly stated the long-term plan is to **train the entire network from scratch on real data** (see `from-scratch-intent` memory) — the pretrained init exists only to speed up iteration during the design-phase test programs.

---

## 6. Test-program depth — three scripts

**Chosen:** Three runnable scripts in [scripts/](../../scripts/) that build on the same shared package:

- `test_01_forward_shapes.py` — forward pass, assert intermediate shapes
- `test_02_backward.py` — forward + MSE + `optimizer.step()`, verify gradients
- `test_03_overfit.py` — overfit a single fixed sample to ~0 loss

**Alternatives considered:** any individual one of the three alone (would have skipped useful regression coverage). Picked all three so the suite stays as a smoke test as the pipeline grows.

**Revisit:** This is a permanent suite; new test programs (transformer decoder, multi-view) get their own numbered scripts.
