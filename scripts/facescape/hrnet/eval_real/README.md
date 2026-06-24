# Real-image evaluation (sim-to-real test)

Evaluate the FaceScape-trained HRNetV2-W18 models on **real** face data. This is
the Phase3/Phase5 step: the gap between synthetic-val NME and real-image NME is
the finding. No model or eval code is written here -- we reuse the HRNet repo's
`tools/test.py` and only convert each real dataset into the 300W CSV format.

## Models under test (in `output/hrnet/`)

| tag | training | checkpoint |
|---|---|---|
| `forte` | WITH background aug + more data | `output/hrnet/forte_trained/300W/face_alignment_facescape_w18/model_best.pth` |
| `sharp` | NO background, less data | `output/hrnet/sharp_trained/model_best.pth` |

## Pieces

- `build_aflw2000_csv.py` -- AFLW2000-3D `.mat` -> `data/AFLW2000/test_yaw45.csv` (teaching scaffold).
- `face_alignment_aflw2000_w18.yaml` -- test config (DATASET: 300W, TESTSET -> that CSV).
- `build_wflw_csv.py` -- WFLW 98-pt `.txt` -> `data/wflw/test_wflw68.csv` (98->68 iBUG remap; reuses the AFLW framing fn for parity).
- `face_alignment_wflw_w18.yaml` -- WFLW test config (also DATASET: 300W; remapped to 68-pt).

## Setup (once)

```bash
.venv/bin/pip install scipy        # needed to read AFLW2000 v5 .mat files
```

## Run (from repo root)

```bash
# 1. build the CSV (after filling the scaffold gaps)
.venv/bin/python scripts/facescape/hrnet/eval_real/build_aflw2000_csv.py

# 2. eval each model -- prints  nme / [008] / [010]
HR=third_party/HRNet-Facial-Landmark-Detection
for M in \
  output/hrnet/forte_trained/300W/face_alignment_facescape_w18/model_best.pth \
  output/hrnet/sharp_trained/model_best.pth ; do
  echo "=== $M ==="
  PYTHONPATH=$HR .venv/bin/python scripts/facescape/hrnet/eval_real/run_eval.py \
    --cfg scripts/facescape/hrnet/eval_real/face_alignment_aflw2000_w18.yaml \
    --model-file "$M"
done
```

The NME line is `compute_nme()` with 68-pt inter-ocular normalization
(`||pts[36]-pts[45]||`), directly comparable to published 300W/AFLW numbers.

## WFLW (second real dataset)

Download from https://wywu.github.io/projects/LAB/WFLW.html (ungated) into `data/WFLW/`:
`WFLW_images.tar.gz` -> `data/WFLW/WFLW_images/...`, `WFLW_annotations.tar.gz` ->
`data/WFLW/WFLW_annotations/.../list_98pt_rect_attr_test.txt`. Then:

```bash
# 1. build the 68-pt CSV (full WFLW test set; add --drop_large_pose to mimic AFLW's yaw<=45)
.venv/bin/python scripts/facescape/hrnet/eval_real/build_wflw_csv.py

# 2. eval each model on WFLW
HR=third_party/HRNet-Facial-Landmark-Detection
for M in \
  output/hrnet/forte_trained/300W/face_alignment_facescape_w18/model_best.pth \
  output/hrnet/sharp_trained/model_best.pth ; do
  echo "=== $M ==="
  PYTHONPATH=$HR .venv/bin/python scripts/facescape/hrnet/eval_real/run_eval.py \
    --cfg scripts/facescape/hrnet/eval_real/face_alignment_wflw_w18.yaml \
    --model-file "$M"
done
```

Note WFLW is a harder benchmark than AFLW2000 (more occlusion/blur/large-pose), so
expect a higher NME even for a well-trained model -- compare the sim-to-real *gap*,
not the absolute number, against the AFLW result. Before trusting the number, eyeball
one overlay to confirm the 98->68 remap (iBUG 36/45 = outer eye corners).
