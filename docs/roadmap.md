# Multi-view Head Pose Tracking — Roadmap

Live project checklist. Update the checkboxes as work progresses. Dates are absolute
($1$ Aug $2026$, etc.); the original draft used `8/1` / `9/1` shorthand.

## Literature & replication

- [ ] Continue literature review of ML-based multi-view tracking papers
- [ ] Replicate at least one existing paper using their code and dataset
  - [ ] Train on Great Lakes (HPC) to validate our workflow

## Codebase selection & adaptation

- [ ] Select one existing codebase to adapt
- [ ] Adapt existing code to work with the 3DWF dataset
  - [ ] Remove final layers from ResNet input to produce a latent space
- [ ] Determine how to handle the depth channel in RGB-D input
  - [ ] Option: skip ResNet and pass depth directly to the transformer
  - [ ] Option: add depth as a fourth channel for ResNet
  - [ ] Find another paper that does this

## Our own dataset

- [x] Generate a virtual RGB-D multi-view set from FaceScape (synthetic precursor:
  textured TU render → RGB-D + world point clouds + landmarks + 6-DoF pose). Tooling
  and pipeline in [docs/data/facescape_rgbd.md](data/facescape_rgbd.md); pilot = ~10
  publishable subjects, neutral, 5 views. **Renderer complete and eyeball-verified
  2026-06-10** ([scripts/facescape/render_views.py](../scripts/facescape/render_views.py)).
- [ ] Generate our own test dataset
  - [ ] Try using the average of head pose from RetinaFace views as ground truth
  - [ ] Try putting a tracking helmet on a model head as ground truth
  - [ ] Try putting a model head on the UR5e as ground truth

## Dataset validation & modeling (plan set 2026-06-10)

Architecture invariant for all models below: **CNN ResNet backbone → transformer
decoder** (keep this feeding order throughout).

- [x] Extend the renderer to vary **lighting** across the synthetic set. Randomized
  (seeded) domain randomization: per view, geometry is rendered once and RGB is
  rendered under `--n_lights` random lighting conditions (1-3 frontal-hemisphere
  directional lights, random direction/intensity/color-temp + ambient). Output:
  `rgb__NN.png` per view, lighting spec logged in `meta.json`, `lighting_panel.png`
  for eyeballing. Camera configs were already variable via `cameras.json`/`default_ring`.
  - [ ] (Optional refinement) Vary **light color temperature**. Sample a black-body
    temperature in $[3000, 8000]$ K and convert to RGB (Tanner-Helland approximation)
    rather than arbitrary RGB, so the lit side picks up a physical warm/cool tint
    while shadow/ambient stays neutral. Add a `color` field to the `Light` dataclass
    and a `kelvin_to_rgb` helper in `sample_lighting`. Deferred: lights stay white
    until the direction/intensity loop is verified; cheap fallback is global
    white-balance jitter in the torch Dataset.
- [ ] Train a **known-working model (e.g. RetinaFace)** on the synthetic dataset with the
  varied lighting/camera configs — a baseline to confirm the data is learnable.
- [ ] Train on **Great Lakes (HPC)** and evaluate the result to **validate the dataset**.
- [ ] Adapt **MVGFormer** to do **face detection** (the multi-view path).

## Modeling & evaluation

- [ ] Try making our own model
- [ ] Test the model on phantoms
- [ ] Test the model on humans

## Publication

- [ ] If it works, start drafting the paper by **2026-08-01**
- [ ] If targeting ICRA, send full paper draft to Mark by **2026-09-01**
