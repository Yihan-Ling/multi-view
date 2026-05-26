# MvP & MVGFormer — study notes for multi-view RGB-D head pose

> Reference study to inform the v0 pipeline (per-view face crop $\to$ ResNet-50 + transformer decoder $\to$ 3D face landmarks $\to$ head-model fit $\to$ 6-DoF pose).

## 0. Reading guide

This document studies two reference codebases:

- **MvP** (`mvp/`) — NeurIPS 2021 "Multi-view Pose Transformer", Wang et al., for multi-person 3D body pose on CMU Panoptic.
- **MVGFormer** (`MVGFormer/`) — CVPR 2024 "Multiple View Geometry Transformers", Liao et al., extending MvP with iterative query refinement and explicit triangulation.

Both target multi-person 3D body pose. Our project targets single-face, multi-view RGB-D head pose. The value of this study is therefore **architecture and patterns**, not application code:

- Projective attention (how 3D queries sample 2D image features under known camera geometry).
- Transformer decoder design for geometric problems.
- Multi-view fusion strategies.
- End-to-end differentiable triangulation.

**How to read:**

1. Skim §1 (MvP) for the foundational ideas. We will NOT build MvP (§1.10 explains why).
2. Read §2 (MVGFormer) in detail — this is the candidate codebase to build on later.
3. §3 is a side-by-side comparison.
4. §4 is the actionable porting checklist for adapting MVGFormer to our task. Read this last but most carefully.
5. §5 lists open questions to confirm with stakeholders before any porting work.

Repo paths in this doc are relative to `/home/carson/Documents/Work/IGMR/`. File citations are of the form `path/file.py:LINE`.

---

## 1. MvP (NeurIPS 2021)

### 1.1 What it solves & why it matters here

MvP regresses multi-person 3D body pose directly from $N$ calibrated RGB views. Its key contribution is **projective attention**: instead of explicitly fusing per-view features into a 3D volume (as VoxelPose did), it lets transformer queries that *carry* a 3D position project themselves into each view's feature map and gather samples there. This is geometrically grounded but learned end-to-end, which is exactly the pattern our multi-view face landmark problem benefits from.

### 1.2 Repo layout

```
mvp/
  configs/                  YAML configs per dataset
  data/                     dataset roots (downloaded separately)
  lib/
    core/                   config loading, loss functions, train/val loops
    dataset/                Panoptic, Shelf, Campus loaders
    models/
      multi_view_pose_transformer.py  main model
      mvp_decoder.py                  decoder layer + projection logic
      pose_resnet.py                  backbone
      matcher.py                      Hungarian matcher
      position_encoding.py            sine + ray positional encodings
      ops/                            custom CUDA deformable conv
        modules/projattn.py
        functions/deform_func.py
        setup.py
    smpl/                   SMPL body model (auxiliary)
    utils/                  cameras, transforms, vis
  models/                   pretrained weights live here
  run/train_3d.py           training entry
  run/validate_3d.py        evaluation entry
```

### 1.3 End-to-end data flow

Entry point: `mvp/run/train_3d.py:193-194` instantiates the model via `models.multi_view_pose_transformer.get_mvp()`.

Forward pass (`mvp/lib/models/multi_view_pose_transformer.py:295-461`):

1. **Input** — multi-view RGB tensor `views: List[Tensor]` (per camera).
2. **Backbone** — concat views along batch dim, push through PoseResNet-50, get multi-scale feature pyramid `all_feats` (`multi_view_pose_transformer.py:297-298`).
3. **Camera ray encoding** — compute per-pixel camera ray direction or 2D coord embedding per feature level (`:312-327`), using intrinsics/extrinsics from `meta`.
4. **Query construction** — build learnable query embeddings of shape `(num_instance * num_keypoints, d_model)`, split into `query_embed` (positional) and `tgt` (content) (`:356-374`).
5. **Reference points** — initial normalized 3D query positions $\hat{P} \in [0,1]^3$ predicted by an MLP from query features (`:389`).
6. **Decoder** — `MvPDecoder` stacks `num_decoder_layers` `MvPDecoderLayer`s, each doing self-attention + projective attention + FFN (`mvp_decoder.py:285-345`).
7. **Output heads** — per-query classification (object / no-object via sigmoid focal loss) and per-query 3D pose regression MLP (`multi_view_pose_transformer.py:401-443`).
8. **Loss** — `SetCriterion` runs Hungarian matching, then sums weighted classification, per-joint L1, per-bone L1, and per-view reprojection L1 losses (`:452-459`).

### 1.4 Backbone

`mvp/lib/models/pose_resnet.py:109-217`. PoseResNet-50 = standard ResNet-50 (Bottleneck blocks, expansion=4) with the final FC replaced by **three transposed-convolution layers** that upsample features back toward the input resolution:

```
input (H, W, 3)
  -> 7x7 conv s2 + BN + ReLU + maxpool s2     # /4
  -> layer1..4 (residual)                     # /4, /8, /16, /32
  -> deconv x3 (stride 2 each)                # /32 -> /16 -> /8 -> /4
  -> 1x1 conv -> heatmaps (only in 2D pretraining)
```

The MvP forward (`pose_resnet.py:198-216`) returns **intermediate features from the deconv stack** at levels indexed by `cfg.NETWORK.use_feat_level` (typically `[0, 1, 2]`, giving 3 progressively higher-resolution maps). Each map has ~256 channels.

**Pretraining:** loaded from `pose_resnet50_panoptic.pth.tar` (CMU Panoptic). The backbone is normally **frozen** during MvP training; only the decoder + queries are learned (`mvp/run/train_3d.py:92-95`).

### 1.5 Transformer decoder — query design

In `mvp_decoder.py`, with definitions also in `multi_view_pose_transformer.py:147-184`:

- **Number of queries:** `num_instance * num_keypoints`. E.g. 10 persons × 15 joints = 150 queries. Multi-person scenes are handled by giving every (person, joint) combination its own query; Hungarian matching at loss time assigns predicted person-slots to ground-truth people.
- **Query embedding schemes** (`multi_view_pose_transformer.py:159-173`):
  - `person_joint` — sum of separate person embedding + joint embedding.
  - `image_person_joint` — also adds an image (camera) embedding.
  - `per_joint` — one embedding per (person, joint) outright.
- **Reference points** $\hat{P}_q \in [0,1]^3$ — predicted from query features via `self.reference_points = nn.Linear(d_model, 3)` (`:389`). Optionally adapted from globally pooled backbone features (`:379-387`).
- **Positional encoding:** `PositionEmbeddingSine` for 2D feature maps; camera ray embedding (`get_rays_new`) concatenated with sampled features inside projective attention.

### 1.6 Projective attention (MvP's signature contribution)

`mvp/lib/models/ops/modules/projattn.py:42-177`. The mechanism:

For each query $q$ with normalized 3D reference point $\hat{P}_q$ and each camera view $v$ with intrinsics $\mathbf{K}_v$ and extrinsics $[\mathbf{R}_v \mid \mathbf{t}_v]$:

1. **Denormalize** $\hat{P}_q \in [0,1]^3 \to P_q \in \mathbb{R}^3$ (in mm), via `norm2absolute()` (`mvp_decoder.py:174-204`).
2. **Project** to image coordinates:
   $$\mathbf{p}_{2D, v, q} = \frac{1}{z_v} \mathbf{K}_v [\mathbf{R}_v \mid \mathbf{t}_v] [P_q; 1]$$
   then apply the image affine transform (for crop/resize) and clamp to image bounds.
3. **Per-head, per-level learnable offsets:** $\delta_{v,q,h,l,k} = \mathrm{Linear}(\text{query}_q)$ produces $K$ deformable sampling offsets around the projected point, per head $h$, per feature level $l$.
4. **Attention weights:** $w_{v,q,h,l,k} = \mathrm{softmax}(\mathrm{Linear}(\text{query}_q))$ across $(l, k)$.
5. **Sample features** via the custom CUDA deformable-conv kernel `DeformFunction.apply()` (`projattn.py:174-175` calling `lib/models/ops/functions/deform_func.py:34-65`) — bilinear sampling at $\mathbf{p}_{2D, v, q} + \delta$ across the feature pyramid.
6. **Ray-aware embedding:** the sampled features are augmented with the camera ray direction at $\mathbf{p}_{2D, v, q}$ via a `rayconv` linear layer.

The output per query per view is a $d_\text{model}$-dim vector. Multi-view fusion (§1.7) then collapses the $v$ dimension.

**Mathematically** the per-query output is:
$$\mathrm{ProjAttn}(q, v) = \sum_{h=1}^{H} \mathbf{W}_h \sum_{l=1}^{L} \sum_{k=1}^{K} w_{v,q,h,l,k}\, \mathrm{BilinearSample}\bigl(F_v^{(l)},\, \mathbf{p}_{2D,v,q} + \delta_{v,q,h,l,k}\bigr)$$

### 1.7 Multi-view fusion

Where: end of each `MvPDecoderLayer.forward()` (`mvp_decoder.py:223-276`), after projective attention has produced per-view features `feat: (B, V, Q, C)`.

Modes (selected by config):

| Mode | Operation |
|---|---|
| `mean` | $\frac{1}{V}\sum_v$ |
| `cat_proj` | concatenate along channel, then linear back to $d_\text{model}$ |
| `attn_fuse_dot_prod` | learned per-view weights from query–view dot product |
| `attn_fuse_subtract` | residual-from-mean reweighting |

A **validity mask** zeros out features for queries whose projected $(u,v)$ falls outside the view's image bounds (`mvp_decoder.py:193-221`).

### 1.8 Loss & training

`mvp/lib/models/multi_view_pose_transformer.py:208-232` + `mvp/lib/core/loss.py`:

- **Classification:** sigmoid focal loss, binary object / no-object per query (`loss.py:55-84`). $\alpha=0.25, \gamma=2$.
- **Per-joint L1:** $\sum_q m_q \|P_q - P_q^{\text{gt}}\|_1$, optionally MPJPE-style L2 (`loss.py:81-116`).
- **Per-bone L1:** L1 on bone-vector predictions (defined by skeleton edges).
- **Per-projection L1:** L1 on 3D-pose-reprojected 2D positions across views (geometric consistency).
- **Cardinality (logged only):** L1 between predicted person count and GT.

**Hungarian matching** (`mvp/lib/models/matcher.py:20-81`) constructs a bipartite assignment between the 150 queries and the variable-cardinality GT persons, with cost = $\lambda_1 \cdot \text{cls cost} + \lambda_2 \cdot \text{L1 pose}$. After matching, the per-query losses are computed only on matched pairs.

Optimizer: AdamW with differential LR (lower for projective-attention sampling layers).

### 1.9 Key files inventory

| File | Purpose |
|---|---|
| `mvp/lib/models/multi_view_pose_transformer.py` | Main model, query init, output heads, `SetCriterion` |
| `mvp/lib/models/mvp_decoder.py` | `MvPDecoderLayer` (self-attn + proj-attn + FFN), view-fusion modes |
| `mvp/lib/models/ops/modules/projattn.py` | Projective attention module |
| `mvp/lib/models/ops/functions/deform_func.py` | Autograd wrapper around the custom CUDA op |
| `mvp/lib/models/ops/setup.py` | Build script for the CUDA op |
| `mvp/lib/models/pose_resnet.py` | PoseResNet-50 backbone with deconv head |
| `mvp/lib/models/matcher.py` | Hungarian matcher |
| `mvp/lib/core/loss.py` | All loss implementations |
| `mvp/run/train_3d.py` | Training entry, distributed setup |

### 1.10 Why we are NOT building MvP

The codebase is 2021-era and has rotted in several ways:

1. **Custom CUDA op** (`mvp/lib/models/ops/setup.py:50-91`) hardcodes compute capabilities `sm_60` through `sm_89`. Won't build on H100 (`sm_90`) or newer without patching. Import statement `import Deformable as DF` (`deform_func.py:31`) crashes silently if the build fails.
2. **Legacy `mmcv`** — uses `mmcv.runner.get_dist_info()` (`train_3d.py:52`), which is from old mmcv (pre-2.0 API).
3. **`tensorboardX`** (`requirements.txt:10`) instead of native `torch.utils.tensorboard`.
4. **No version pins** in `requirements.txt` — implicit reliance on a specific (old) PyTorch.
5. **Hardcoded dataset assumptions** — joint counts, dataset-specific conversions are baked into `multi_view_pose_transformer.py:434-441`.

MVGFormer inherits the same projective-attention CUDA op but is overall newer and maintained more recently (CVPR 2024, last update Mar 2025 per README), so it's the right base if we want production code.

---

## 2. MVGFormer (CVPR 2024)

### 2.1 What it adds over MvP, in one paragraph

MvP regresses 3D poses in a single decoder pass — queries project, sample, refine features, and the final output of the last layer is the prediction. MVGFormer instead makes the decoder an **iterative geometry-aware refinement loop**: at each layer, queries project to 2D, sample features, predict a 2D offset per view, and then **explicit (differentiable) triangulation** recovers a refined 3D position that becomes the reference point for the next layer. The math of triangulation — not just attention — is in the gradient graph, which is what gives MVGFormer its much better generalization across camera setups (per its CVPR paper).

### 2.2 Repo layout — delta vs MvP

```
MVGFormer/
  configs/                  YAML configs
  data/                     dataset roots
  docs/                     extra install/usage docs   <-- new
  lib/
    core/                   like MvP but with PerProjectionL1Loss2D added
    dataset/                Panoptic, Shelf, Campus
    models/
      dq_transformer.py     DyanmicQueryTransformer (replaces MvP top-level model)
      dq_decoder.py         DQDecoderLayer (iterative loop + triangulation)
      multi_view_pose_transformer.py  base class still present, inherited from
      mvp_decoder.py        base decoder still present
      pose_resnet.py        unchanged backbone
      ops/modules/projattn.py  projective attention WITH rayconv variants
    mvn/                    <-- new: multi-view-net utilities
      utils/multiview.py    triangulation (DLT linear, batched, weighted)
      models/, datasets/    less central
    structural/             <-- new: bone-length-constrained triangulation
      structural_triangulation.py
      adapter.py
    smpl/                   SMPL body model
    utils/                  cameras, transforms
  process/                  <-- new: data processing scripts
  scripts/                  <-- new: launch / experiment scripts
  run/train_3d.py           training entry
  run/validate_3d.py        eval entry
  run/generate_video.py     <-- new: qualitative output
  tpose.pt                  T-pose template for query_adapt_center init
```

### 2.3 End-to-end data flow

Entry: `MVGFormer/run/train_3d.py:246-250` selects the transformer class via `eval('models.' + config.TRANSFORMER + '.get_mvp')(...)`. Typical config: `TRANSFORMER: 'dq_transformer'`.

Top-level model: `DyanmicQueryTransformer` (`MVGFormer/lib/models/dq_transformer.py:120`), built by `get_mvp()` at `dq_transformer.py:756-771`. Inherits a lot from `MultiviewPosetransformer` (MvP's top-level class).

`DyanmicQueryTransformer.forward()` (`dq_transformer.py:335-599`):

1. **Backbone extract** (`:350-372`):
   ```python
   all_feats = self.backbone(torch.cat(views, dim=0), self.use_feat_level)
   ```
   Multi-level features at each `use_feat_level` index.
2. **Query embeddings** (`:391-432`) built from `joint_embedding(num_keypoints)` + `instance_embedding(num_instance)`, expanded to batch.
3. **Initialize 3D reference points** (`:437-479`) — strategy chosen by `cfg.DECODER.init_method`:
   - `query_adapt` — MLP predicts initial 3D from query features.
   - `query_adapt_center` — predicts center + uses T-pose template (`tpose.pt`) offset per joint.
   - `sample_space` — uniform-sample 3D space.
   - `gt_noise` — Gaussian-perturbed ground truth (training only, debug).
4. **Decoder iteration** (`:541-562`):
   ```python
   hs, inter_references, inter_references_2d, ... = self.decoder(
       tgt, reference_points, src_flatten_views, meta=meta, ...)
   ```
   Returns intermediate outputs from every decoder layer (for auxiliary losses).
5. **Output heads** (`:599-620`) — per-layer classification + 3D pose; the final layer's output is the prediction at eval time.

`DQDecoder.forward()` (`dq_decoder.py:1107-1172`) is just a loop over layers (verified, line 1135):

```python
for lid, layer in enumerate(self.layers):
    reference_points_input = reference_points[:, :, None]
    ...
    output, reference_points, ref_points_2d, projs_2d_absolute, outputs_class = layer(
        output, query_pos_in, reference_points_input, src_views, ...
    )
    if self.return_intermediate:
        intermediate.append(output)
        intermediate_reference_points.append(reference_points)
        ...
```

The real work is `DQDecoderLayer.forward()` (`dq_decoder.py:850-1045`) — covered in §2.5.

### 2.4 Backbone

Same `pose_resnet.py` as MvP. The difference is that **MVGFormer uses all `use_feat_level` levels actively** in `ProjAttn` (multi-scale deformable sampling), whereas MvP's released configs typically use a single level.

Pretrained weights still expected at `${POSE_ROOT}/models/pose_resnet50_panoptic.pth.tar`. Backbone frozen during training (`MVGFormer/run/train_3d.py:118-121`).

### 2.5 Iterative query refinement — the core idea

`DQDecoderLayer.forward()` (`dq_decoder.py:850-1045`) does one iteration:

```
# Stage 1: project queries into each view, gather features
attn_feature_views, ref_points_expand_views = self.generate_features(
    src_views, tgt, query_pos, src_padding_mask, reference_points,
    meta, src_spatial_shapes, level_start_index
)                                            # dq_decoder.py:875

# Stage 2: update query features (self-attn + view fusion + FFN)
tgt_update = self.update_feature(tgt, attn_feature_views, query_pos)  # :882

# Stage 3: classify + filter low-confidence queries before triangulation
outputs_class = self.class_embed(tgt_update)                         # :889
batch_ids, query_ids = self.generate_valid_masks(
    outputs_class_prob, method=self.query_filter_method, value=threshold
)                                                                    # :904

# Stage 4: predict per-view 2D offsets + per-view confidences
refined_2d_poses_absolute, confidences, projs_2d_poses_abs, ... =
    self.calculate_2d_offsets(nviews, device, attn_feature_views,
                              ref_points_expand_views, rgb_views)    # :938

# Stage 5: differentiable triangulation -> new 3D reference points
# (uses confidences as per-view weights for DLT; method ∈ batch / linalg /
#  default / cpu / st / st-gt)
new_reference_points = learnable_triangulate(
    refined_2d_poses_absolute, confidences, ...,
    method=self.triangulation_method
)
```

The next decoder layer receives `new_reference_points` as its 3D queries, so refinement compounds across the 4 layers. **All steps are differentiable** — the triangulation SVD is implemented in torch (`MVGFormer/lib/mvn/utils/multiview.py`) so gradients flow through it.

> **Note on a hidden constant:** `dq_decoder.py:1141` does `query_num = round(query_num_mul_joints / 15)` — the body-joint count (15 for CMU Panoptic) is **hardcoded in the loop**, not just in configs. Important for porting (§4.2).

### 2.6 Projective attention with ray embeddings — diff vs MvP

`MVGFormer/lib/models/ops/modules/projattn.py:42-220`. The class is again `ProjAttn(d_model, n_levels, n_heads, n_points, projattn_posembed_mode)`.

`projattn_posembed_mode` selects how the projected position is encoded (verified at `projattn.py:82-89`):

```python
if projattn_posembed_mode == 'use_rayconv':
    self.rayconv = nn.Linear(d_model + 3, d_model)   # +3 = camera ray direction
elif projattn_posembed_mode == 'use_2d_coordconv':
    self.rayconv = nn.Linear(d_model + 2, d_model)   # +2 = 2D image coord
elif projattn_posembed_mode == 'ablation_not_use_rayconv':
    self.rayconv = nn.Linear(d_model, d_model)       # no geometric encoding
```

Default and recommended is `use_rayconv`: at the projected pixel, the camera ray $\mathbf{r}_{v,q} \in \mathbb{R}^3$ (unit vector from camera center through the pixel) is concatenated to the sampled feature, then projected back to $d_\text{model}$.

Forward (`projattn.py:115-220`):

1. Compute `sample_grid = clamp(reference_points * 2 - 1, -1.1, 1.1)` to map normalized coords to `F.grid_sample`'s `[-1, 1]` range (`:134`).
2. Per feature level, `F.grid_sample(src_views[lvl], sample_grid[:, :, lvl:lvl+1, :], align_corners=False)` extracts features at projected (u,v) (`:140-143`).
3. Concatenate features with camera ray embeddings, pass through `rayconv`.
4. Compute learnable sampling offsets `sampling_offsets: Linear(d_model, n_heads*n_levels*n_points*2)` and attention weights, then apply the deformable-conv CUDA op (same kernel as MvP, inherited from Deformable DETR).

The CUDA op compilation is at `MVGFormer/lib/models/ops/setup.py` — the build command (per `MVGFormer/README.md:67-70`) is:

```bash
cd ./lib/models/ops
CUDA_HOME=/usr/local/cuda-11.0/ python setup.py build install
```

### 2.7 Multi-view geometry modules (what's new vs MvP)

#### 2.7.1 DLT triangulation — `MVGFormer/lib/mvn/utils/multiview.py`

Functions (line numbers approximate from agent reading):

- `triangulate_point_from_multiple_views_linear_torch()` — single-point DLT.
- `triangulate_batch_of_points()` — batched.
- `triangulate_batch_of_points_batch_version()` — alternative batched implementation with optional confidence weighting.

Math: for a 3D point $\mathbf{X} = [X, Y, Z, 1]^\top$ observed in view $i$ at $\mathbf{x}_i = [u_i, v_i, 1]^\top$ with projection matrix $\mathbf{P}_i = \mathbf{K}_i [\mathbf{R}_i \mid \mathbf{t}_i] \in \mathbb{R}^{3 \times 4}$:

$$\begin{pmatrix} u_i\, \mathbf{p}_i^{3\top} - \mathbf{p}_i^{1\top} \\ v_i\, \mathbf{p}_i^{3\top} - \mathbf{p}_i^{2\top} \end{pmatrix} \mathbf{X} = \mathbf{0}$$

where $\mathbf{p}_i^{j\top}$ is the $j$-th row of $\mathbf{P}_i$. Stacking across $N$ views yields $\mathbf{A} \mathbf{X} = \mathbf{0}$ with $\mathbf{A} \in \mathbb{R}^{2N \times 4}$. Solve

$$\mathbf{X} = \arg\min_{\|\mathbf{X}\|=1} \|\mathbf{A}\mathbf{X}\|^2$$

via SVD: $\mathbf{X}$ is the right singular vector corresponding to the smallest singular value.

When per-view confidences $c_i \in [0, 1]$ are provided, the rows of $\mathbf{A}$ are scaled by $c_i$, which down-weights uncertain detections.

#### 2.7.2 Structural triangulation — `MVGFormer/lib/structural/`

`structural_triangulation.py` + `adapter.py` (`adapter.py:20-91`, `structural_triangulate_points()`). Enforces **skeletal kinematic constraints** during triangulation: instead of triangulating each joint independently, jointly optimize all joints in a person so that

1. Their 2D reprojections match observations.
2. Pairwise bone-length distances $\|J_a - J_b\|$ match a learned or supplied reference skeleton.

Methods exposed by `cfg.DECODER.triangulation_method`:

| Method | Behavior |
|---|---|
| `batch` | Vectorized DLT, no skeletal constraint |
| `linalg` | Same idea, different implementation |
| `default` | Sequential DLT |
| `cpu` | CPU implementation (fallback) |
| `st` | Structural triangulation with bone-length constraint |
| `st-gt` | `st` using GT bone lengths (training-time supervision) |

#### 2.7.3 Iterative undistortion — `dq_decoder.py:119-204`

Implements OpenCV-style iterative undistortion inside the decoder:

For radial coefficients $(k_1, k_2, k_3)$ and tangential $(p_1, p_2)$, given a distorted normalized image point $(x_d, y_d)$, iterate (typically 5 steps):

$$r^2 = x^2 + y^2 \qquad i_{\text{cdist}} = \frac{1}{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}$$

$$\Delta X = 2 p_1 xy + p_2 (r^2 + 2x^2) \qquad \Delta Y = p_1 (r^2 + 2y^2) + 2 p_2 xy$$

$$x_{\text{new}} = (x_d - \Delta X) \cdot i_{\text{cdist}} \qquad y_{\text{new}} = (y_d - \Delta Y) \cdot i_{\text{cdist}}$$

This is done in PyTorch so it stays in the gradient graph during triangulation.

### 2.8 Triangulation methods — which to use when

| Method | Speed (training) | Differentiable | Uses bone prior | Notes |
|---|---|---|---|---|
| `batch` | Fast | Yes | No | Default for early experiments |
| `linalg` | Fast | Yes | No | Alternative; same outputs |
| `default` | Slow | Yes | No | Reference; debug |
| `cpu` | Very slow | Yes | No | Fallback / sanity |
| `st` | Medium | Yes | Yes (learned) | Better with strong prior |
| `st-gt` | Medium | Yes | Yes (GT) | Training only — bakes oracle |

For our face-landmark use, see §4.5.

### 2.9 Loss & training

Same loss family as MvP plus:

- **`PerProjectionL1Loss2D`** (`MVGFormer/lib/core/loss.py`, ~line 245) — supervises the **per-view 2D offsets** predicted inside each decoder layer (before triangulation). This gives a much stronger 2D signal than just reprojection-of-3D, which is critical because the 2D offsets are what drive triangulation.

Auxiliary losses are computed at every decoder layer's output (not just the last), summed with weights from `weight_dict`. This is the standard DETR-family trick — supervising intermediates stabilizes deep transformer decoders.

Optimizer (verified pattern from MvP, same in MVGFormer): AdamW with two LR groups:

- **Base LR** for query embeddings, FFNs, classification head.
- **Base LR $\times$ `cfg.DECODER.lr_linear_proj_mult`** (typically $0.1$) for the projective-attention sampling-offset layer and the reference-point regressor. These layers are sensitive and benefit from a slower LR.

### 2.10 Query initialization strategies — which is used when

`cfg.DECODER.init_method` (handled in `dq_transformer.py:437-479`):

| Method | When to use |
|---|---|
| `query_adapt` | Generic; safe default |
| `query_adapt_center` | When you have a meaningful template (T-pose, mean face); predicts only the center + adds per-joint offsets from the template |
| `sample_space` | Cold-start / generalization eval |
| `gt_noise` | Debugging — verifies the rest of the pipeline can refine from "almost correct" |

For face landmarks, a **mean face template** is the natural analog of `tpose.pt` → use `query_adapt_center` with a learned/fixed mean face.

### 2.11 Key files inventory

| File | Purpose |
|---|---|
| `MVGFormer/run/train_3d.py` | Entry, distributed setup, LR scheduler |
| `MVGFormer/lib/models/dq_transformer.py` | `DyanmicQueryTransformer` main model, query construction, ref point init |
| `MVGFormer/lib/models/dq_decoder.py` | `DQDecoderLayer` iteration body, undistortion |
| `MVGFormer/lib/models/ops/modules/projattn.py` | Projective attention w/ rayconv variants |
| `MVGFormer/lib/mvn/utils/multiview.py` | DLT triangulation + Camera class |
| `MVGFormer/lib/structural/structural_triangulation.py` | Bone-constrained triangulation |
| `MVGFormer/lib/structural/adapter.py` | Wrapper exposing `st` and `st-gt` to the decoder |
| `MVGFormer/lib/core/loss.py` | Losses (adds `PerProjectionL1Loss2D`) |
| `MVGFormer/lib/dataset/JointsDataset.py` | Dataset base — image+camera+joints loader, affine tracking |
| `MVGFormer/lib/models/pose_resnet.py` | Backbone (unchanged from MvP) |

### 2.12 Buildability today

From `MVGFormer/README.md` (lines 51-80) and `requirements.txt`:

- **Python:** 3.10 (`conda create -n mvgformer python==3.10`).
- **`mmcv-full`:** installed via `mim install mmcv-full`. Notoriously sensitive to PyTorch+CUDA version alignment.
- **CUDA op compilation** (deformable attention) requires `CUDA_HOME` set, e.g. `CUDA_HOME=/usr/local/cuda-11.0/ python setup.py build install`. Will need patching of `sm_*` targets for very new GPUs.
- **Other deps:** `torch`, `torchvision`, `smplx`, `h5py`, `scipy`, `numpy`, `matplotlib`, `tensorboardX`, `wandb` — all stable.

**Risk assessment for building today:** moderate. The CUDA op is the main hazard; the rest is conventional. README was last updated Mar 2025 (per the changelog at the top), so it's plausibly still build-able on cu11/cu12 + recent PyTorch.

---

## 3. Comparison — MvP vs MVGFormer

| Aspect | MvP (NeurIPS 2021) | MVGFormer (CVPR 2024) |
|---|---|---|
| Top-level model | `MultiviewPosetransformer` | `DyanmicQueryTransformer` (subclasses MvP's) |
| Decoder structure | Self-attn + ProjAttn + FFN, $L$ layers, predict at end | Same + per-layer 2D-offset prediction + triangulation, predict at every layer |
| Reference point update | Implicit (via deformable offsets, but no explicit 3D update between layers) | **Explicit triangulation produces new 3D ref points per layer** |
| Geometric prior in gradient graph | Only via projection (read-only) | Triangulation SVD is differentiable; gradients flow through geometry |
| Query init | Random / learned MLP from queries | Same options + `query_adapt_center` with T-pose template |
| Backbone use | Single feature level (typical) | Multi-level (`use_feat_level = [0,1,2]`) |
| Positional encoding in ProjAttn | 2D sine / camera ray (basic) | `use_rayconv` adds learned linear over (feature ⊕ camera-ray direction) |
| Multi-view fusion | `mean` / `cat_proj` / `attn_fuse_*` inside decoder | Same modes, plus **triangulation acts as a fusion mechanism for 3D** |
| Loss family | Focal-cls + per-joint L1 + per-bone L1 + per-projection L1 | Same + `PerProjectionL1Loss2D` on 2D offsets per layer (aux supervision) |
| Hungarian matching | Yes (multi-person) | Yes (multi-person; query filtering by confidence added before triangulation) |
| Generalization across camera setups | Moderate (relies on learned ProjAttn to figure out geometry) | Strong (explicit geometry generalizes by construction) |
| Build today | Painful (sm_* pins, legacy mmcv) | Moderate (mmcv version sensitivity + CUDA op) |

**Bottom line:** MVGFormer = MvP + explicit differentiable triangulation in the decoder loop + better feature fusion (rayconv) + per-layer supervision. For a problem like ours where camera calibration is known precisely, the explicit-geometry approach is the right pattern.

---

## 4. Porting MVGFormer to multi-view RGB-D face landmark + 6-DoF head pose

Format: each axis lists the file(s) involved, the action tag (REUSE / MODIFY / REPLACE / DROP), and a one-line rationale. This is a checklist for a future porting session, not an implementation plan.

### 4.1 Input modality — RGB to RGB-D

| What | Where | Action | Notes |
|---|---|---|---|
| First conv layer | `MVGFormer/lib/models/pose_resnet.py` (~`:116`, `Conv2d(3, 64, ...)`) | MODIFY | Either change to `Conv2d(4, 64, ...)` for stacked RGBD, or keep two backbones (one RGB, one D) and fuse in decoder |
| Image loader | `MVGFormer/lib/dataset/JointsDataset.py:85-96` | MODIFY | Currently `cv2.imread(..., IMREAD_COLOR)`. Need to load depth alongside (depth file format / units / handling of invalid pixels is a design choice) |
| Image normalization | `JointsDataset.py:105-106` | MODIFY | Add depth normalization (e.g., divide by camera's max-range constant; mask invalid pixels to zero or to mean depth) |
| Pretrained ResNet | `pose_resnet.py:220-230` (state-dict load) | MODIFY | The pretrained backbone is 3-channel. For 4-channel: either (a) replicate channel-1 (red) weights to channel-4 and fine-tune, (b) initialize channel-4 randomly with small variance, or (c) keep backbone 3-channel and inject depth as a cross-attention key in the decoder |

### 4.2 Task — multi-person body joints to single-face landmarks

| What | Where | Action | Notes |
|---|---|---|---|
| `num_keypoints` | config + `multi_view_pose_transformer.py:147` | MODIFY | 15 (Panoptic) → face landmark count (68 / 51 / 478 — TBD, §5) |
| **Hardcoded `15` in decoder loop** | `MVGFormer/lib/models/dq_decoder.py:1141` (`query_num = round(query_num_mul_joints / 15)`) | MODIFY | Replace with `cfg.NETWORK.NUM_JOINTS`. Easy to miss — flagged because it's in code, not config |
| `num_instance` | config | MODIFY | 1024 (max persons) → 1 (single cropped face) |
| Query embedding scheme | `multi_view_pose_transformer.py:149, 159-184` | MODIFY | `'person_joint'` becomes effectively `'per_joint'` since `num_instance = 1`. Simpler and smaller |
| Classification head (object/no-object) | `multi_view_pose_transformer.py:193-195, 583-627` | DROP | No detection problem — every query is a known landmark |
| Hungarian matcher | `multi_view_pose_transformer.py:217-222`, `mvp/lib/models/matcher.py` | DROP | Single subject + known landmark indices = direct supervision per query |
| Cardinality loss | `multi_view_pose_transformer.py:630-650` | DROP | Always 1 face per crop |
| Query filtering before triangulation | `dq_decoder.py:889-908` (`outputs_class_prob`, `generate_valid_masks`) | MODIFY → simplify | Without classification, can drop the filter and triangulate all queries; or replace with a "visibility" filter based on per-view crop validity |

### 4.3 Per-view face crop integration

| What | Where | Action | Notes |
|---|---|---|---|
| Pre-loader face detection + crop | (NEW) wrap `JointsDataset.__getitem__` or add a transform | NEW | Detect face in each full view, select user-chosen face (across views), crop RGBD tile. Track the crop affine in `affine_trans` |
| Affine tracking | `JointsDataset.py:143-161` (`affine_trans`, `inv_affine_trans`) | REUSE / extend | Already designed for crop+resize affines; extend the affine to compose face-crop + resize |
| `IMAGE_SIZE` config | e.g. `MVGFormer/configs/panoptic/knn5-lr4-q1024-g8.yaml:43-45` | MODIFY | `[960, 512]` → tight face crop, e.g. `[256, 256]` or `[512, 512]` |
| Calibration update after crop | `MVGFormer/lib/mvn/utils/multiview.py:23-31` (`Camera.update_after_crop()`) | REUSE | Already handles principal-point shift; just feed it the face-crop affine |

### 4.4 Multi-person to single-person — 3D space reframing

| What | Where | Action | Notes |
|---|---|---|---|
| `MULTI_PERSON.SPACE_SIZE` | config (~`:81-94`) | REPLACE | `[8000, 8000, 2000]` mm (scene) → face region (~`[300, 300, 300]` mm centered on face) |
| `SPACE_CENTER` | config | REPLACE | World scene center → estimated face center (from detection / depth) per frame, OR a fixed origin in a face-aligned coordinate frame |
| `MAX_PEOPLE_NUM` | config (~`:94`) | DROP / set to 1 | |
| Reference-point MLP | `multi_view_pose_transformer.py:389` | REUSE | Output is normalized to `[0,1]^3` of the space — works regardless of space dimensions |

### 4.5 Geometry modules

| What | Where | Action | Notes |
|---|---|---|---|
| `ProjAttn` | `MVGFormer/lib/models/ops/modules/projattn.py` | REUSE | Geometry-agnostic; works on any 3D query |
| DLT triangulation | `MVGFormer/lib/mvn/utils/multiview.py` | REUSE | Same SVD-based approach |
| Structural triangulation `st` / `st-gt` | `MVGFormer/lib/structural/*` | REUSE — but redefine "bones" | Face landmarks have a meaningful structure (eyes pair, mouth contour, jaw chain); define `LIMBS_FACE` analogous to `LIMBS15` in `mvp/lib/core/loss.py:152-154` |
| Undistortion in decoder | `dq_decoder.py:119-204` | REUSE | Standard radial+tangential distortion model |

### 4.6 Camera calibration plumbing

| What | Where | Action | Notes |
|---|---|---|---|
| Calibration loader | `JointsDataset.py:186-220` | REUSE if rig output matches `(fx, fy, cx, cy, R, T)` schema | Write a small adapter from our rig's calibration format to this schema |
| Distortion coefficients | `JointsDataset.py:217` and `Camera` class | MODIFY | MVGFormer's distortion path exists; if our RGB-D rig is rectified, can pass zeros |
| Per-frame extrinsics | already supported | REUSE | Useful if cameras move (e.g., on a robot) |

### 4.7 Depth-channel exploitation — three options

The four-channel-input option (§4.1) is the most natural, but depth could also be injected later in the pipeline:

| Option | Where it attaches | Cost | Benefit |
|---|---|---|---|
| (a) **4th input channel to backbone** | `pose_resnet.py:116` | Loses some pretraining; need careful init | End-to-end learnable, no extra modules |
| (b) **Depth as geometric-consistency loss** | new term added in `dq_decoder.py` after triangulation, or in `core/loss.py` | Need to unproject depth maps and handle occlusions | Decouples feature extraction from depth; depth becomes a regularizer |
| (c) **Direct depth unprojection per landmark** | Post-decoder, in a new module that takes refined 2D landmarks + depth map and produces a 3D estimate | Extra computational stage; needs alignment of depth and color | Uses depth as a direct 3D signal, anchors scale absolutely |

**Recommended starting point:** option (a) with channel-replicated init, plus option (b) as an auxiliary loss with small weight (e.g., $\lambda_{\text{depth}} = 0.1$). Option (c) is a strong fallback if (a)+(b) underperform — and is also a great sanity-check baseline.

### 4.8 Loss / supervision

| What | Where | Action | Notes |
|---|---|---|---|
| `PerJointL1Loss` (3D) | `mvp/lib/core/loss.py:81-100` (and MVGFormer's equivalent) | REUSE | Identical for landmarks |
| `PerProjectionL1Loss2D` | `MVGFormer/lib/core/loss.py` (~`:245`) | REUSE | Critical for the iterative-refinement scheme |
| `PerBoneL1Loss` | `lib/core/loss.py` | MODIFY | Redefine `LIMBS` for face landmark connectivity (e.g., mouth contour, eye contour) |
| Classification loss `loss_ce` | `multi_view_pose_transformer.py` | DROP | See §4.2 |
| Cardinality loss | same | DROP | See §4.2 |
| Per-view visibility flag | `JointsDataset.py:110, 164-178` (`joints_2d_vis`, `joints_3d_vis`) | MODIFY → enhance | Set visibility = 0 per (view, landmark) when the landmark is self-occluded in that view (profile views occlude the off-side ear, jaw) |
| `weight_dict` | config | MODIFY | Remove `loss_ce`, `loss_cardinality`. Keep + tune `loss_pose_perjoint`, `loss_pose_perprojection_2d`, `loss_pose_perbone` |
| **NEW** 6-DoF loss | see §4.9 | NEW | Geodesic + L2 |

### 4.9 Head-model fitting — NEW additive stage

This stage does not exist in MVGFormer. Two options for `lib/models/head_pose_from_landmarks.py` (new file in our repo):

**Option A — closed-form rigid alignment.** Given predicted 3D landmarks $\hat{L} \in \mathbb{R}^{K \times 3}$ and template landmarks $L_0 \in \mathbb{R}^{K \times 3}$ from FLAME / BFM / a custom mean head, solve the orthogonal Procrustes problem for $(R, t, s)$:

$$\min_{R \in SO(3), t \in \mathbb{R}^3, s > 0} \sum_{k=1}^{K} \|\hat{L}_k - s R L_{0,k} - t\|_2^2$$

Closed-form via SVD of $\sum_k (\hat{L}_k - \bar{\hat{L}})(L_{0,k} - \bar{L_0})^\top$. Differentiable in PyTorch.

**Option B — learned MLP head.** $f_\theta: \mathbb{R}^{K \times 3} \to \mathrm{SE}(3)$, output rotation as 6D representation (Zhou et al. 2019) plus 3D translation, trained with the loss below.

**6-DoF loss:**

$$\mathcal{L}_{6\text{DoF}} = \lambda_R \cdot \text{geo}(R_{\text{pred}}, R_{\text{gt}}) + \lambda_t \cdot \|t_{\text{pred}} - t_{\text{gt}}\|_2$$

where the geodesic on $SO(3)$ is

$$\text{geo}(R_1, R_2) = \arccos\!\left(\frac{\mathrm{tr}(R_1^\top R_2) - 1}{2}\right)$$

Start with Option A (no extra params, immediate sanity check). Add Option B if Procrustes residuals are large relative to landmark noise.

**Integration point:** post-decoder, before `SetCriterion`. Append `outputs['pred_pose_6dof']` and add the loss term to `weight_dict`.

### 4.10 Biggest risks

1. **Loss of ResNet pretraining when adding the 4th channel.** Mitigation: replicate channel weights, freeze backbone for the first few epochs while the rest of the network learns, then unfreeze. Have a 3-channel-only baseline running in parallel as a sanity check.
2. **Crop affine propagation bugs corrupt the entire geometry.** A 1-pixel error in tracking the face-crop affine becomes a degree-scale error in 3D after triangulation. Mitigation: write a unit test that takes a synthetic 3D point, runs it through (project $\to$ crop $\to$ resize $\to$ inverse) and asserts round-trip recovery. Run this test continuously during development.
3. **Scale ambiguity with tight face crops and short-baseline rigs.** Face landmarks within a $\sim$200 mm cube observed from $\lesssim$0.5 m baselines have weakly-conditioned triangulation. Mitigation: depth channel (§4.7c) anchors scale; head-model prior (§4.9, scale $s$) regularizes; or constrain $s$ to a narrow range around a known mean.
4. **The hardcoded `15` in `dq_decoder.py:1141`** is the kind of thing a quick port will miss — explicitly listed here so it doesn't bite.
5. **Triangulation degenerate cases.** If a face landmark is visible in only 1 view (self-occlusion), DLT fails. Need per-landmark per-view confidences feeding into the triangulation weights, and need to handle the "<2 views visible" case explicitly (skip triangulation, fall back to the previous layer's 3D estimate, or use depth).

---

## 5. Open questions to confirm before any porting

Before any actual porting work begins, the following decisions affect the architecture and configs:

1. **Number of face landmarks** — 68 (dlib / 300W), 51, 98 (WFLW), 478 (MediaPipe-style dense)? Affects query count, loss compute, and what "bones" mean.
2. **Head model choice** — FLAME, BFM, custom from in-house head scans? Affects whether §4.9 Option A is straightforward (mature model + template) or needs work.
3. **Camera rig specs** — number of views $N$, baseline length, intrinsics format (OpenCV vs ROS vs custom), distortion model. Determines triangulation conditioning and the calibration adapter.
4. **Source of ground-truth 3D landmarks for supervision** — synthetic from a head model? In-house captures with multi-view fitting? Existing public RGB-D face datasets (BIWI, Pandora)? Determines training feasibility entirely.
5. **Realtime constraints** — does this need to run at $\geq$30 Hz on a specific GPU? Affects choice of triangulation method (§2.8) and number of decoder layers.
6. **Multi-face vs single-face** — pipeline stage 2 already crops a *user-selected* face. Is selection always upstream of the model, or do we ever want the model itself to handle multiple faces (and thus keep classification + Hungarian)?

These should be resolved (at least provisionally) before committing to the porting work. Several of them have load-bearing implications on the architecture (e.g., 6 above can resurrect the classification head we just dropped in §4.2).
