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

- [x] Extend the renderer to vary **lighting** across the synthetic set
  (eyeball-passed 2026-06-12). `--lighting` flag: geometry is rendered once, then RGB
  is rendered under 4 random frontal-hemisphere directional lights (random
  direction/intensity + ambient; **not seeded** — reproducibility dropped). Each
  condition is world-consistent across all cameras. Output: `rgb_1..rgb_4.png`
  alongside `rgb.png` per view, plus a `lighting_panel.png` grid for eyeballing.
  Camera configs were already variable via `cameras.json`/`default_ring`.
  - [ ] (Optional refinement) Vary **light color temperature**. Sample a black-body
    temperature in $[3000, 8000]$ K and convert to RGB (Tanner-Helland approximation)
    rather than arbitrary RGB, so the lit side picks up a physical warm/cool tint
    while shadow/ambient stays neutral. Add a `color` field to the `Light` dataclass
    and a `kelvin_to_rgb` helper in `sample_lighting`. Deferred: lights stay white
    until the direction/intensity loop is verified; cheap fallback is global
    white-balance jitter in the torch Dataset.
- [ ] **HPC dry-run:** run a known-working **RetinaFace** repo on the synthetic data
  end-to-end on **Great Lakes** — shakes out the HPC env, data loading, multi-GPU,
  checkpointing, and format compatibility. NOTE: this is a *workflow* validation only;
  its detection score saturates and does **not** test the landmark/pose GT, so it is
  not the dataset's scientific validation. (RetinaFace's lasting role = off-the-shelf
  face cropper, used pretrained, not trained on synthetic.)
- [ ] **Real dataset validation:** train the **day-1 landmark/pose regressor** (ResNet-50
  → deconv head for 68 landmarks + MLP head for 6-DoF pose) and report **landmark error
  (px/mm) + pose error (deg) on a subject-disjoint held-out split**. This exercises the
  GT we care about and is a direct precursor to MVGFormer.
- [ ] Adapt **MVGFormer** to do **face detection** (the multi-view path).

## Modeling & evaluation

- [ ] Try making our own model
- [ ] Test the model on phantoms
- [ ] Test the model on humans

## Publication

- [ ] If it works, start drafting the paper by **2026-08-01**
- [ ] If targeting ICRA, send full paper draft to Mark by **2026-09-01**
