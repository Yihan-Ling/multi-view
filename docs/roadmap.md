# Multi-view Head Pose Tracking — Roadmap

Live project checklist. Update the checkboxes as work progresses. Dates are absolute
($1$ Aug $2026$, etc.); the original draft used `8/1` / `9/1` shorthand.

> **Pivot (2026-06-16):** the active line of work is now **sim-to-real landmark
> transfer** — train HRNetV2-W18 on FaceScape synthetic renders to predict 68-pt 2D
> landmarks (RGB only) and test on real, ungated benchmarks (AFLW2000-3D, WFLW).
> Closing the synthetic→real gap is the current scientific question; the multi-view
> RGB-D pose pipeline below remains the longer-term goal that this de-risks.

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
