#!/bin/bash
# Run ONCE on a Great Lakes LOGIN node (not inside a job) to build the HRNet
# training env. Miniconda + the env live on the TURBO drive so they survive,
# don't eat your small home quota, and are visible from the compute nodes.
#
#   ssh greatlakes
#   bash /nfs/turbo/coe-igmr-pub/yhling/multi-view/scripts/facescape/hrnet/greatlakes_setup_env.sh
#
# torch.cuda.is_available() will print False here -- that's normal, login nodes
# have no GPU. Verify CUDA inside an salloc allocation (see greatlakes_README.md).
set -euo pipefail

# --- EDIT if your turbo folder differs ----------------------------------------
TURBO_DIR="${TURBO_DIR:-/nfs/turbo/coe-igmr-pub/yhling}"
ENV_NAME="${ENV_NAME:-hrnet}"
# cu124 wheels run on the A40 (spgpu) nodes. Bump if a node driver demands newer.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
# -----------------------------------------------------------------------------

CONDA_ROOT="$TURBO_DIR/miniconda3"

if [ ! -d "$CONDA_ROOT" ]; then
  echo ">> installing miniconda into $CONDA_ROOT"
  mkdir -p "$TURBO_DIR"
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O "$TURBO_DIR/miniconda.sh"
  bash "$TURBO_DIR/miniconda.sh" -b -p "$CONDA_ROOT"
  rm -f "$TURBO_DIR/miniconda.sh"
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"

# Recent miniconda requires accepting the Anaconda channel ToS before `conda create`.
# Harmless if already accepted. (|| true: older conda has no `tos` subcommand.)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

if ! conda env list | grep -qE "/$ENV_NAME\$"; then
  echo ">> creating conda env '$ENV_NAME'"
  conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"

echo ">> installing python deps"
pip install --upgrade pip
pip install torch torchvision --index-url "$TORCH_INDEX"
# HRNet lib + facescape_aug + (scipy/matplotlib only needed for later eval)
pip install numpy pandas opencv-python Pillow tensorboardX yacs hdf5storage scipy matplotlib

python -c "import torch, torchvision, cv2, tensorboardX, yacs, hdf5storage; \
print('torch', torch.__version__, '| cuda build', torch.version.cuda, \
'| cuda avail (False on login node is OK):', torch.cuda.is_available())"

echo
echo ">> done. To use this env later:"
echo "   source $CONDA_ROOT/etc/profile.d/conda.sh && conda activate $ENV_NAME"
