#!/bin/bash
# Great Lakes batch job: from-scratch + photometric + background + EYE-LEAK +
# RETINAFACE CROP. Twin of greatlakes_train_eyeblack_bg.sbatch, pointed at the
# HRNet_train_retina_eyeleak_bg bundle via
# face_alignment_facescape_w18_scratch_retina.yaml.
# Submit from the repo root ON TURBO:
#   cd /nfs/turbo/coe-igmr-pub/yhling/multi-view
#   sbatch scripts/facescape/hrnet/greatlakes_train_retina.sbatch
#
#SBATCH --job-name=hrnet_fs_retina
#SBATCH --partition=spgpu
#SBATCH --account=mdraelos0
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8          # matches cfg WORKERS=8
#SBATCH --mem-per-gpu=48G         # request the full A40; more is useless w/o more GPUs
#SBATCH --time=06:00:00           # 60 ep BS64 ~2.5-3h; padded. Undershoot = no extra cost
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yhling@umich.edu
#SBATCH --output=log/slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# --- env (built once by greatlakes_setup_env.sh) -------------------------------
TURBO_DIR="${TURBO_DIR:-/nfs/turbo/coe-igmr-pub/yhling}"
source "$TURBO_DIR/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-hrnet}"

# --- sanity: bundle present (gitignored -> must be rsynced, not cloned) ---------
# NOTE: the bundle is built LOCALLY (RetinaFace crops baked into the CSV) and
# rsynced whole to turbo -- the detector does NOT run on the cluster.
test -f data/facescape/HRNet_train_retina_eyeleak_bg/train.csv \
  || { echo "MISSING data/facescape/HRNet_train_retina_eyeleak_bg/ -- rsync the retina bundle to turbo first"; exit 1; }

mkdir -p log
echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# --- train ---------------------------------------------------------------------
python scripts/facescape/hrnet/run_train.py \
  --cfg scripts/facescape/hrnet/face_alignment_facescape_w18_scratch_retina.yaml
