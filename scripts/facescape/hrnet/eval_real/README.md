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
- `build_wflw_csv.py` -- TODO (second dataset; 98->68 iBUG remap).

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
  PYTHONPATH=$HR .venv/bin/python $HR/tools/test.py \
    --cfg scripts/facescape/hrnet/eval_real/face_alignment_aflw2000_w18.yaml \
    --model-file "$M"
done
```

The NME line is `compute_nme()` with 68-pt inter-ocular normalization
(`||pts[36]-pts[45]||`), directly comparable to published 300W/AFLW numbers.
