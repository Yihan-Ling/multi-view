# Multi-view Head Pose Tracking — Roadmap

Live project checklist. Update the checkboxes as work progresses. Dates are absolute
($1$ Aug $2026$, etc.); the original draft used `8/1` / `9/1` shorthand.

> **Pivot (2026-06-16):** the active line of work is now **sim-to-real landmark
> transfer** — train HRNetV2-W18 on FaceScape synthetic renders to predict 68-pt 2D
> landmarks (RGB only) and test on real, ungated benchmarks (AFLW2000-3D, WFLW).
> Closing the synthetic→real gap is the current scientific question; the multi-view
> RGB-D pose pipeline below remains the longer-term goal that this de-risks.

> **Update (2026-07-13):** iteration 1 of the **early-fusion multi-view 3D-landmark
> model** is **trained and the depth ablation is complete** — see
> [Iteration 1 below](#iteration-1--early-fusion-multi-view-3d-landmarks-active-2026-07).
> Result: RGB-D $\approx 3.47$–$3.51$ mm vs RGB-only $3.55$ mm held-out MPJPE — depth
> gives **no measurable benefit** on the clean, well-conditioned 5-view synthetic set
> (delta is within the $\sim 0.05$ mm run-to-run noise floor; both arms hit the same
> $\sim 3.5$ mm data-limited floor). **Next track: "messy data" — apply the same
> sim-to-real domain randomization that worked for HRNet (random-background
> compositing + blur + a randomized camera ring) to robustly prove the model works
> under realistic, varied conditions rather than an easy fixed-rig in-domain set.** See
> [Iteration 2 below](#iteration-2--robustness-under-messy-data-active-2026-07).

## Current active track: HRNet sim2real landmark transfer

Metric = inter-ocular NME on real images. Train = FaceScape synthetic renders
(HRNetV2-W18, from scratch, BS64, schedule [30,50], 60 epochs). Key lever to date =
photometric/background augmentation. Build adapter:
[scripts/facescape/build_hrnet_landmark_dataset.py](../scripts/facescape/build_hrnet_landmark_dataset.py)
(CLI: `--composite-prob`, `--out-root`, `--fill-holes`). Eval driver:
[scripts/facescape/hrnet/eval_real/run_eval.py](../scripts/facescape/hrnet/eval_real/run_eval.py).

### Ablation status

| Condition | Adds | Synth-val NME | Real AFLW2000 | Real WFLW |
| --- | --- | --- | --- | --- |
| sharp | no aug | — | — | $0.477$ |
| forte | baked bg | — | — | $0.497$ |
| **photo_aug** | photometric + bg + **eye-leak** | $\sim 0.043$ | **$0.146$** | **$0.187$** |
| iter2_nobg | photometric only (no bg) | $0.041$ | $0.233$ | $0.324$ |
| iter3_eyeblack_bg | bg + flat-black eyes (leak fixed) | $0.039$ | $0.196$ | $0.269$ |

**photo_aug is still the best model.** Both later conditions regressed (eval'd
2026-06-30). Note the inverted relationship between synth-val and real NME: iter2/iter3
have *better* synth-val yet *worse* real NME — synth-val is not a reliable proxy.

- [x] Photometric augmentation → $\sim 3.4\times$ improvement over sharp/forte
  (real AFLW2000 $0.49/0.41 \rightarrow 0.146$). Gap narrowed, not closed.
- [x] Fix **eye-leak bug**: TU mesh has no eyeballs, so the eye region is a
  `depth==0` hole and the random background leaked *through* the eyes;
  `composite_over_bg` now `binary_fill_holes` → paints interior holes black.
- [x] Train iter2_nobg (photometric only) and iter3_eyeblack_bg on Great Lakes
  (`model_best.pth` present for both).
- [x] Run the 4 pending real evals (eval'd 2026-06-30). Findings:
  - **Background compositing helps a lot** — iter2_nobg (no bg) is worst by a wide
    margin. Keep bg randomization.
  - **The eye-leak was inadvertently good augmentation** — flat-black eyes (iter3)
    regressed vs the bg-through-eyes behavior (photo_aug). The eye region wants
    *varied texture*, not a flat fill. Restore the photo_aug eye behavior as default.

### Diagnosis (photo_aug, 2026-06-25)

Error is **contour-dominated**: jaw/outline NME $0.30$–$0.35$, while the inner face
transfers well ($\sim 0.08$). Likely root cause = **cap / no-hair renders**, not
framing. Eye-region NME $\sim 0.082$ ($\sim 7.4$ mm), normalizer = outer eye-corner
distance (lm36–lm45).

### Planned next iterations (priority order, post-eval)

- [ ] **RetinaFace-crop training** (iteration 2 / next experiment). Re-crop the
  synthetic training set with the same RetinaFace regime used on real test images.
  Doubles as a **confounder-removal**: if contour error persists after matching crop
  framing, the cause is genuinely the renders; if it drops, it was framing.
- [ ] **Render realism** — address the dominant contour gap directly (hair / remove
  the bald cap silhouette). This is the highest-value lever per the diagnosis. Fold
  a *physically plausible* eye treatment in here rather than a flat fill.
- [ ] (Optional) **Randomized eye-hole fill** — turn the accidental eye-leak into a
  controlled augmentation: fill interior holes with random texture/color per image
  instead of a constant. **eye-white is rejected** — the iter3 result shows flat fills
  regress; another flat color would share the failure mode.

## Longer-term goal: multi-view RGB-D head pose pipeline

Architecture invariant: **CNN ResNet backbone → transformer decoder** (keep this
feeding order throughout).

### Iteration 1 — early-fusion multi-view 3D landmarks (ACTIVE, 2026-07)

An MVGFormer-style pipeline that stops at 3D landmarks (no 6-DoF yet), with
**early fusion** of depth (RGB-D as a 4-channel ResNet input). Build plan:
[docs/early_fusion_iter1_buildplan.md](early_fusion_iter1_buildplan.md).
**Complete: trained end-to-end and the depth ablation is run.** Headline — depth as a
4th input channel is *neutral* on clean well-conditioned synthetic; this motivates the
messy-data robustness track (Iteration 2).

- [x] Phase 0 — camera geometry: `project` + differentiable DLT `triangulate`
  (round-trip test exact to $\sim 10^{-11}$).
- [x] Phase 1 — multi-view RGB-D dataset (`virtual_camera_data`, 246 subjects,
  5 views; schema-aware loader; geometric visibility; reprojection exact to $0$ px).
- [x] Phase 2 — mean-face template from the **FaceScape bilinear model** (average of
  847 subjects, neutral expression); metric query box from the data.
- [x] Phase 3 — 4-channel RGB-D ResNet-50 + 3-deconv backbone, multi-view wired
  ($(B,N,4,H,W) \to (B,N,256,128,128)$).
- [x] Phase 4 — projective attention (project queries → grid-sample → mean-fuse).
- [x] Phase 5 — decoder $\times 4$ (self-attention, 2D-offset + confidence heads,
  confidence-weighted triangulation). **Keystone overfit test: one sample
  $41.3 \to 0.95$ mm, per-layer refinement confirmed.**
- [x] Phase 6 — deep-supervised losses (3D L1 + visibility-masked 2D L1).
- [x] Phase 7 — training script (subject-disjoint split, from scratch), GPU-verified
  end-to-end; Great Lakes sbatch ready ([scripts/greatlakes_early_fusion.sbatch](../scripts/greatlakes_early_fusion.sbatch)).
- [x] **First full training** (Great Lakes A40, RGB-D) → best held-out MPJPE
  $\approx 9.98$ mm at bs8/40ep. Required three numerical-stability fixes first (NaN
  blow-up at ep12): signed floor on the perspective-divide depth
  ([multi_view/decoder.py](../multi_view/decoder.py)), signed floor on the DLT
  dehomogenize ([multi_view/geometry.py](../multi_view/geometry.py)), and grad-norm
  clipping + skip-non-finite-grad guard in the train loop.
- [x] **Moved to laptop** (RTX 4060, bs2/img256) — the job is small ($\sim 14$ s/epoch),
  so GL was overkill. Smaller batch (4$\times$ more grad steps/epoch) + longer cosine
  dropped held-out MPJPE $\approx 3\times$: **3.47 mm @ 60 ep, 3.38 mm @ 100 ep** (RGB-D).
  Epochs past $\sim 80$ give diminishing returns — **data-limited floor $\sim 3.5$ mm**
  on 197 train subjects. Established a **run-to-run noise floor $\sim 0.05$ mm** (CUDA
  nondeterminism: identical recipe gave 3.47 vs 3.51 mm).
- [x] **RGB-D vs RGB-only ablation** (60 ep each, same laptop/bs): RGB-only **3.55 mm**
  vs RGB-D $\approx 3.49$ mm → **depth is neutral here** — the $\sim 0.06$ mm delta is
  within the noise floor. Consistent with theory: the 5-view rig is already
  well-conditioned (triangulation nails depth without a sensor), clean renders leave RGB
  no localization headroom for depth to add, and both arms hit the same data-limited
  floor. Implication: depth's payoff needs *weak* geometry / degraded inputs → Iteration 2.
- [x] **Metric logging** added to [scripts/train_early_fusion.py](../scripts/train_early_fusion.py):
  each run writes `metrics.csv` (`epoch,lr,train_loss,val_loss,val_mpjpe,skipped,sec`)
  and `train.log`; `evaluate()` now also reports validation loss.
- [ ] (Optional) Multi-seed (0/1/2) per arm to statistically confirm "depth neutral"
  (RGB-D mean inside RGB-only spread).

Key differences vs MVGFormer: single face + fixed 68 landmark queries (no detection /
Hungarian / NMS / `SPACE_SIZE`), absolute metric coordinates, early depth fusion, and
a simplified pure-PyTorch projective attention (single-point `grid_sample`, mean view
fusion, single feature scale, no rayconv / structural triangulation / undistortion —
the deferred future-work list). See
[docs/references/mvp_mvgformer_study.md](references/mvp_mvgformer_study.md).

### Iteration 2 — robustness under messy data (ACTIVE, 2026-07)

**Motivation.** Iteration 1 was trained and evaluated on a *clean, fixed-rig* synthetic
set — same 5 camera viewpoints every subject, black background, sharp renders. That is an
easy in-domain setting. To **robustly prove the model actually works** (not just memorizes
an easy distribution) and to move toward real deployment, apply the **same sim-to-real
domain randomization that gave the HRNet landmark model a $\sim 3.4\times$ real-image
improvement**: random backgrounds, blur, and — the multi-view-specific addition — a
**randomized camera ring** so the model is not tied to one fixed rig.

Design principle: augment the **RGB inputs only**; the depth channel and the 3D/2D
**ground-truth labels stay clean**, so MPJPE remains an honest metric. Applied to **both
train and val** (train = fresh randomness each epoch; val = fixed per-sample for a stable
metric).

- [x] **Background + blur augmentation** — `multi_view/data/augment.py`
  (`AugConfig` + `MultiViewAugmentor`): per-view random-background compositing (indoor
  photos, keyed off the `binary_fill_holes(depth>0)` silhouette so nothing leaks through
  the eyeless eye-holes) + Gaussian blur. Wired into the Dataset (`augmentor`,
  `aug_deterministic`) and the train script (`--bg-prob`, `--blur-prob`,
  `--blur-sigma-max`). Verified on real data (RGB changes; depth + GT untouched; val
  deterministic); preview `scratch/iter2_aug_preview.png`.
- [x] **Randomized camera ring** — `render_views.py` gained `random_ring()` (per-subject
  random radius / FOV / azimuth arc / elevation, 5 views fixed for batching) + a
  `--rand_ring` flag and `--out_root`/`--data_root` CLI. Removes the fixed-rig limitation.
- [ ] **Re-render** the 246-subject set with `--rand_pose --rand_ring` to a new dir
  (`virtual_camera_random_ring`), preserving the fixed-rig set. **Needs the EGL render env
  (`.venv-data`), which is not on the laptop — run where that env lives (desktop).**
- [ ] **Train + report** the RGB-D model on the messy data (random ring + bg + blur),
  compare held-out MPJPE to the clean $\sim 3.5$ mm baseline. Success = stays accurate and
  generalizes under realistic messy conditions.

**Note:** later short-baseline / fewer-view experiments overlap the deferred
**depth scale-anchor** idea and the teammate's **late-fusion** track — coordinate so the
experiments don't collide.

**This week (objectives):** (1) eyeball the bg+blur preview; (2) run a bg+blur training
pass on the existing fixed-rig data (decoupled from the re-render); (3) re-render the
random-ring set on the desktop; (4) train + report on the fully messy data. Training runs
on the laptop; only rendering needs the EGL env.

### Literature & replication

- [ ] Continue literature review of ML-based multi-view tracking papers
- [ ] Replicate at least one existing paper using their code and dataset
  - [x] Validate the Great Lakes (HPC) workflow — done via the HRNet runs above
    (acct mdraelos0, spgpu A40, conda+repo on `/nfs/turbo/coe-igmr-pub/yhling`).

### Codebase selection & adaptation

- [ ] Select one existing codebase to adapt
- [ ] Adapt existing code to work with the 3DWF dataset
  - [ ] Remove final layers from ResNet input to produce a latent space
- [ ] Determine how to handle the depth channel in RGB-D input
  - [ ] Option: skip ResNet and pass depth directly to the transformer
  - [ ] Option: add depth as a fourth channel for ResNet
  - [ ] Find another paper that does this

### Our own dataset

- [x] Generate a virtual RGB-D multi-view set from FaceScape (synthetic precursor:
  textured TU render → RGB-D + world point clouds + landmarks + 6-DoF pose). Tooling
  in [docs/data/facescape_rgbd.md](data/facescape_rgbd.md). **Renderer complete and
  eyeball-verified 2026-06-10**
  ([scripts/facescape/render_views.py](../scripts/facescape/render_views.py)).
- [x] Extend the renderer to vary **lighting** across the synthetic set
  (eyeball-passed 2026-06-12). `--lighting` flag: geometry rendered once, RGB
  rendered under 4 random frontal-hemisphere directional lights; world-consistent
  across cameras. Output: `rgb_1..rgb_4.png` + `lighting_panel.png` per view.
  - [ ] (Optional) Vary **light color temperature** — sample black-body $[3000,8000]$ K
    (Tanner-Helland), add a `color` field to `Light` + `kelvin_to_rgb` in
    `sample_lighting`. Deferred; cheap fallback = white-balance jitter in the Dataset.
- [ ] Generate our own test dataset
  - [ ] Try using the average of head pose from RetinaFace views as ground truth
  - [ ] Try putting a tracking helmet on a model head as ground truth
  - [ ] Try putting a model head on the UR5e as ground truth

### Dataset validation & modeling

- [ ] **Real dataset validation:** train the **day-1 landmark/pose regressor**
  (ResNet-50 → deconv head for 68 landmarks + MLP head for 6-DoF pose) and report
  **landmark error (px/mm) + pose error (deg) on a subject-disjoint split**.
- [ ] Adapt **MVGFormer** to do **face detection** (the multi-view path).

### Modeling & evaluation

- [ ] Try making our own model
- [ ] Test the model on phantoms
- [ ] Test the model on humans

## Publication

- [ ] If it works, start drafting the paper by **2026-08-01**
- [ ] If targeting ICRA, send full paper draft to Mark by **2026-09-01**
