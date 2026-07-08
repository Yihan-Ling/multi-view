#!/bin/bash
# Great Lakes batch job: early-fusion multi-view 3D-landmark model, from scratch.
# Submit from the repo root ON TURBO:
#   cd /nfs/turbo/coe-igmr-pub/yhling/multi-view
#   sbatch scripts/greatlakes_early_fusion.sbatch
#
# For the FIRST run prefer the interactive salloc+tmux flow so you can watch the
# per-epoch MPJPE live (see scripts/facescape/hrnet/greatlakes_README.md). Use
# this batch script for the unattended full run once you trust it.
#
#SBATCH --job-name=earlyfusion_mv
#SBATCH --partition=spgpu
#SBATCH --account=mdraelos0
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G          # full A40
#SBATCH --time=04:00:00            # 40 ep @ bs8/img256 ~1-2h on A40; padded
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yhling@umich.edu
#SBATCH --output=log/slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# --- env (same conda env as the HRNet runs; torch + torchvision + scipy + PIL) --
TURBO_DIR="${TURBO_DIR:-/nfs/turbo/coe-igmr-pub/yhling}"
source "$TURBO_DIR/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-hrnet}"

# --- sanity: data present (8.7G, gitignored -> must be rsynced, not cloned) -----
test -d data/facescape/virtual_camera_data \
  || { echo "MISSING data/facescape/virtual_camera_data -- rsync it to turbo first"; exit 1; }
test -f multi_view/assets/mean_face_68.npy \
  || { echo "MISSING multi_view/assets/mean_face_68.npy -- git pull the latest"; exit 1; }

mkdir -p log output/early_fusion
echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# --- train (RGBD arm). For the RGB-only ablation, add --no-depth and change --out.
python scripts/train_early_fusion.py \
  --root data/facescape/virtual_camera_data \
  --assets multi_view/assets \
  --out output/early_fusion \
  --epochs 40 --bs 8 --lr 1e-4 --img-size 256 --num-layers 4 --workers 8
