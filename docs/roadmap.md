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

- [ ] Generate a virtual RGB-D multi-view set from FaceScape (synthetic precursor:
  textured TU render → RGB-D + world point clouds + landmarks + 6-DoF pose). Tooling
  and pipeline in [docs/data/facescape_rgbd.md](data/facescape_rgbd.md); pilot = ~10
  publishable subjects, neutral, 5 views. Blocked on FaceScape license/download.
- [ ] Generate our own test dataset
  - [ ] Try using the average of head pose from RetinaFace views as ground truth
  - [ ] Try putting a tracking helmet on a model head as ground truth
  - [ ] Try putting a model head on the UR5e as ground truth

## Modeling & evaluation

- [ ] Try making our own model
- [ ] Test the model on phantoms
- [ ] Test the model on humans

## Publication

- [ ] If it works, start drafting the paper by **2026-08-01**
- [ ] If targeting ICRA, send full paper draft to Mark by **2026-09-01**
