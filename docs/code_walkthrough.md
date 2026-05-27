# Code walkthrough

A file-by-file tour of the current repo: what each module does, what each script tests, and what its output means. Intended as an on-ramp for someone reading the code for the first time.

Companion docs:
- [docs/design/decisions.md](design/decisions.md) — the "why" behind each design choice, with alternatives and revisit triggers.
- [docs/references/mvp_mvgformer_study.md](references/mvp_mvgformer_study.md) — the reference architectures (MvP, MVGFormer) the eventual multi-view pipeline draws from.

---

## Project layout

```
multi-view/
├── multi_view/          # the library: model code
│   ├── weight_init.py
│   ├── backbone.py
│   ├── head.py
│   ├── model.py
│   └── __init__.py
├── scripts/             # three runnable test programs
│   ├── _init_paths.py
│   ├── test_01_forward_shapes.py
│   ├── test_02_backward.py
│   └── test_03_overfit.py
└── docs/                # design notes (decisions.md, mvp_mvgformer_study.md)
```

The high-level data flow is:

$$\text{RGBD image } (B, 4, 256, 256) \xrightarrow{\text{backbone}} (B, 256, 64, 64) \xrightarrow{\text{head}} (B, K, 3)$$

where $K = 68$ landmarks and each row is a 3D coordinate.

---

## The library — `multi_view/`

### [multi_view/weight_init.py](../multi_view/weight_init.py)

Holds **one** function: `init_conv1_4ch_from_pretrained(new_conv, pretrained_weight)`.

- A normal ImageNet ResNet-50 has a 3-channel first conv with weight shape $(64, 3, 7, 7)$. We want a 4-channel conv $(64, 4, 7, 7)$ so we can feed in $(R, G, B, D)$.
- The function:
  1. Sanity-checks the new conv is 4-channel and the pretrained tensor is 3-channel.
  2. Copies the pretrained RGB filters into channels 0–2 of the new conv.
  3. Copies the **red** channel filter again into channel 3 (the depth channel).
- Wrapped in `torch.no_grad()` so the copies don't show up in the autograd graph.

The "replicate red" trick is a known shortcut: depth edges look broadly like grayscale edges, and red is a reasonable proxy. See [decisions.md §5](design/decisions.md). The docstring marks this as temporary scaffolding — long-term plan is to train fully from scratch on real RGB-D data.

### [multi_view/backbone.py](../multi_view/backbone.py) — `RGBDPoseResNet50`

A ResNet-50 trunk with two modifications:

**Modification 1 — 4-channel input:**
- Loads `resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)`.
- Saves the original 3-channel conv1 weight.
- Replaces `rn.conv1` with a fresh 4-channel `Conv2d(4, 64, 7, stride=2, padding=3)`.
- Calls `init_conv1_4ch_from_pretrained` to fill the new conv with [RGB | R-copy] filters.

**Modification 2 — drop classifier head, add deconv head (PoseResNet-style):**
- Keeps `conv1 → bn1 → relu → maxpool → layer1..layer4` (i.e. through the last residual stage).
- Drops the original `avgpool` + `fc` (which were for ImageNet classification).
- Appends `self.deconv_head`: 3 × (`ConvTranspose2d(k=4, s=2, p=1)` + `BatchNorm2d` + `ReLU`). Each transposed conv **doubles** the spatial resolution.

Spatial resolution trace for a $256 \times 256$ input:

| Stage | Shape |
|---|---|
| input | $(B, 4, 256, 256)$ |
| conv1 (stride 2) | $(B, 64, 128, 128)$ |
| maxpool (stride 2) | $(B, 64, 64, 64)$ |
| layer1 | $(B, 256, 64, 64)$ |
| layer2 (stride 2) | $(B, 512, 32, 32)$ |
| layer3 (stride 2) | $(B, 1024, 16, 16)$ |
| layer4 (stride 2) | $(B, 2048, 8, 8)$ |
| deconv $\times 3$ | $(B, 256, 64, 64)$ |

So output stride = 4 — input/4 in each spatial dim. The "Simple Baselines for Pose Estimation" paper (Xiao, Wu, Wei, ECCV 2018) introduced this exact pattern; MvP and MVGFormer both port it.

`self.out_channels = 256` is exposed so the head knows what to expect.

#### Aside: what is `ResNet50_Weights.IMAGENET1K_V2`?

It is a **specific set of pretrained weights** for ResNet-50, shipped by torchvision.

- `ResNet50_Weights` — an enum in `torchvision.models` listing every available pretrained checkpoint for ResNet-50.
- `IMAGENET1K` — the weights were trained on **ImageNet-1k** (the classic 1.28M-image, 1000-class classification dataset).
- `V2` — torchvision's **second-generation** training recipe for these weights.

| | `IMAGENET1K_V1` | `IMAGENET1K_V2` |
|---|---|---|
| Recipe | Original 2015-era training (basic aug, SGD, ~76% top-1) | Modern recipe — longer schedule, stronger augmentation (TrivialAugment, RandomErasing), label smoothing, mixup/cutmix, EMA, cosine LR. ResNet-50 hits ~80.86% top-1. |
| When to pick | Reproducing legacy results | Better starting point for transfer learning (almost always preferred today) |

V2 is a strictly-better starting point than V1 for transfer learning. The exact choice does not matter long-term either way, because the pretrained init here is temporary scaffolding (see [decisions.md §5](design/decisions.md)).

### [multi_view/head.py](../multi_view/head.py) — `MLPLandmarkHead`

The deliberately-simplest possible decoder:

1. `AdaptiveAvgPool2d(1)` collapses the $(B, 256, 64, 64)$ feature map to $(B, 256, 1, 1)$.
2. `.flatten(1)` → $(B, 256)$.
3. 2-layer MLP: `Linear(256, 256) → ReLU → Linear(256, 68*3=204)`.
4. `.view(B, 68, 3)` reshape into landmark form.

This intentionally discards spatial information — it is the **regression baseline** that a future DETR-style transformer decoder will be compared against. See [decisions.md §4](design/decisions.md).

### [multi_view/model.py](../multi_view/model.py) — `SingleViewLandmarkModel`

Thin wrapper that wires backbone → head:

```python
backbone = RGBDPoseResNet50(deconv_channels=256, pretrained=True)
head     = MLPLandmarkHead(in_channels=256, num_landmarks=68)
forward(x): return self.head(self.backbone(x))
```

Nothing fancy — this is the "day-1" model that the test scripts exercise.

### [multi_view/__init__.py](../multi_view/__init__.py)

Re-exports `RGBDPoseResNet50`, `MLPLandmarkHead`, `SingleViewLandmarkModel` so the scripts can do `from multi_view import SingleViewLandmarkModel`.

---

## The test scripts — `scripts/`

### [scripts/_init_paths.py](../scripts/_init_paths.py)

Adds the repo root to `sys.path` so the scripts can `import multi_view` without installing the package. Each test script imports it first (with a `# noqa: F401` to suppress the "unused import" lint, since the import is the side effect).

---

### [scripts/test_01_forward_shapes.py](../scripts/test_01_forward_shapes.py) — "do the tensors flow?"

**What it does:**
1. Builds `SingleViewLandmarkModel`, sets `eval()` (no dropout/BN running-stats updates).
2. Creates a random input $x \in \mathbb{R}^{2 \times 4 \times 256 \times 256}$.
3. Under `torch.no_grad()`, asserts three intermediate shapes:
   - `conv1(x)` → $(2, 64, 128, 128)$ — confirms 4-ch conv runs, stride-2 halving works.
   - `backbone(x)` → $(2, 256, 64, 64)$ — confirms the full trunk + deconv head produces stride-4 output with 256 channels.
   - `model(x)` → $(2, 68, 3)$ — confirms the head reshapes to $(B, K, 3)$.

**Expected output:**
```
conv1 output:     (2, 64, 128, 128)
backbone output:  (2, 256, 64, 64)
model output:     (2, 68, 3)
test_01_forward_shapes: PASS
```

**Why this output:** these are the contractually-promised shapes from [decisions.md §3](design/decisions.md). If any assert fails the script raises before printing PASS — so just seeing `PASS` is the whole signal. The shape numbers also let a human eyeball the resolution trace above and confirm strides are correct.

---

### [scripts/test_02_backward.py](../scripts/test_02_backward.py) — "do gradients flow everywhere?"

**What it does:**
1. Seeds RNG (`manual_seed(0)`) so the run is reproducible.
2. Builds model in `train()` mode + `AdamW(lr=1e-3)`.
3. Random input + random target. Forward, computes `F.mse_loss(pred, target)`.
4. Asserts the loss is finite (not NaN/Inf — catches blown-up init).
5. `backward()`, then inspects gradients at **four strategically chosen parameters**:
   - `backbone.conv1.weight` — first layer, **furthest from the loss** (gradient must flow all the way back).
   - `backbone.layer4[-1].conv3.weight` — last residual block in the trunk.
   - `backbone.deconv_head[0].weight` — the first transposed conv (catches if the deconv head detaches).
   - `head.mlp[-1].weight` — last layer, closest to loss.
6. For each, asserts `grad is not None` and `grad.norm() > 0`. Prints each norm.
7. Runs `optimizer.step()` and recomputes loss to show it changed (typically goes down a little, but it is a single step on random targets so not guaranteed to drop).

**Expected output (shape — actual numbers will vary):**
```
  grad backbone.conv1.weight                    norm=X.XXXXe-XX
  grad backbone.layer4[-1].conv3.weight         norm=X.XXXXe-XX
  grad backbone.deconv_head[0].weight           norm=X.XXXXe-XX
  grad head.mlp[-1].weight                      norm=X.XXXXe-XX
loss before step: 1.0XXXXX
loss after step:  ~slightly different from before
test_02_backward: PASS
```

**Why this output:**
- The four grad norms being positive prove there is **no dead branch** in the graph — every important parameter receives gradient signal. A common bug this catches: accidentally wrapping something in `no_grad`, detaching a tensor, or freezing a layer.
- `loss before` near $1.0$ is expected: with random unit-variance pred and target, MSE $\approx \mathrm{Var}(\text{pred}-\text{target}) \approx 2$, but pred passes through layers initialized to keep variance reasonable, so values around 0.8–1.5 are normal.
- `loss after` does not have to be lower — one AdamW step on random targets has no reason to find a good direction. The point is just to show the step ran without exploding.

---

### [scripts/test_03_overfit.py](../scripts/test_03_overfit.py) — "does the model have enough capacity to learn?"

**What it does:**
1. Seeds RNG.
2. Single fixed input ($1 \times 4 \times 256 \times 256$) and a single fixed random target ($1 \times 68 \times 3$).
3. Trains for 200 iterations with AdamW (lr 1e-3), logging every 20 iters.
4. **Success criterion:** final loss $< 10^{-3}$. If not, prints FAIL and `sys.exit(1)`.

**Expected output (shape):**
```
iter    1  loss=1.0XXXXe+00
iter   20  loss=...
iter   40  loss=...
...
iter  200  loss=<below 1e-3>
final loss: X.XXXXXe-XX  (target < 1e-03)
test_03_overfit: PASS
```

A healthy trajectory looks roughly like loss starting near $\sim 1$ and dropping by several orders of magnitude over 200 steps, often crossing $10^{-3}$ comfortably before iteration 200.

**Why this output / what it actually verifies:**
- An overfit-one-sample test is the **classic capacity sanity check**. If you cannot drive loss to $\sim 0$ on a *single* example, something fundamental is broken: vanishing gradients, miscalibrated learning rate, frozen parameters, broken loss, wrong output shape, etc. It is the standard "is the optimization loop even functional" test.
- It is **not** a generalization test — it is the opposite, you *want* to overfit. Generalization gets tested later, on a real dataset.
- The $10^{-3}$ bar is arbitrary but conventional — well below the noise floor of a working model, well above what a broken one can achieve.

If this script fails, the most common culprits (in order) are: a learning rate that is too high causing NaNs, BatchNorm in `eval()` mode somewhere it should not be, or a target tensor that got rebuilt every step.

---

## How the three tests build on each other

| Test | Question answered | Failure means |
|---|---|---|
| 01 forward shapes | Does the architecture have the right plumbing? | Shape bug — wrong stride, wrong channel count |
| 02 backward | Does gradient flow reach every parameter? | Dead branch, detached tensor, frozen layer |
| 03 overfit | Can the optimizer actually drive loss to zero? | Capacity / learning-rate / loss-function bug |

Together they are a **smoke test suite**: cheap, fast, and they catch the most common ways a model gets silently broken as the pipeline grows (transformer-decoder head, multi-view fusion, etc.). [decisions.md §6](design/decisions.md) explicitly calls this out as a permanent suite — new test programs will be added as `test_04_*`, `test_05_*`, etc.
