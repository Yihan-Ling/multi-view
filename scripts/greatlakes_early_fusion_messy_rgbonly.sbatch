#!/bin/bash
# Great Lakes batch job: early-fusion multi-view 3D-landmark model, RGB-ONLY arm,
# trained from scratch on the MESSY baked set (bg + photometric + random-ring).
# Twin of greatlakes_early_fusion.sbatch, pointed at virtual_camera_messy with the
# --no-depth ablation flag. Recipe matches the laptop messy RGB-D run (2.81 mm)
# EXACTLY except for --no-depth, so the only variables are depth on/off + machine
# nondeterminism (batch size kept at 2 to avoid a third confound).
# Submit from the repo root ON TURBO:
#   cd /nfs/turbo/coe-igmr-pub/yhling/multi-view
#   sbatch scripts/greatlakes_early_fusion_messy_rgbonly.sbatch
#
#SBATCH --job-name=ef_messy_rgbonly
#SBATCH --partition=spgpu
#SBATCH --account=mdraelos0
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G          # full A40
#SBATCH --time=08:00:00            # 60 ep @ bs2 on the 3384-folder messy set; padded
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yhling@umich.edu
#SBATCH --output=log/slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# --- env (same conda env as the HRNet runs; torch + torchvision + scipy + PIL) --
TURBO_DIR="${TURBO_DIR:-/nfs/turbo/coe-igmr-pub/yhling}"
source "$TURBO_DIR/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-hrnet}"

# --- sanity: messy data present (23G, gitignored -> must be rsynced, not cloned) --
test -d data/facescape/virtual_camera_messy \
  || { echo "MISSING data/facescape/virtual_camera_messy -- rsync it to turbo first"; exit 1; }
test -f multi_view/assets/mean_face_68.npy \
  || { echo "MISSING multi_view/assets/mean_face_68.npy -- rsync multi_view/ to turbo"; exit 1; }
echo "messy folders: $(ls data/facescape/virtual_camera_messy | wc -l)  (expect 3384)"

mkdir -p log output/early_fusion
echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# --- train (RGB-ONLY arm on messy data; recipe == messy RGB-D 2.81mm run) --------
python scripts/train_early_fusion.py \
  --root data/facescape/virtual_camera_messy \
  --assets multi_view/assets \
  --out output/early_fusion/messy_rgbonly \
  --no-depth \
  --epochs 60 --bs 2 --lr 5e-5 --grad-clip 0.5 --img-size 256 \
  --num-layers 4 --val-frac 0.15 --seed 0 --workers 8
