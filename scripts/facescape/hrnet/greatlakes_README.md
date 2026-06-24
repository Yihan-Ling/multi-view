# Great Lakes runbook — HRNet from-scratch + photometric-aug run

End-to-end steps to run the iteration-1 training (`face_alignment_facescape_w18_scratch.yaml`)
on Great Lakes and pull the model back for the real-image eval. The SSH config
entry (`Host greatlakes`, user `yhling`, ControlMaster) is already in `~/.ssh/config`.

Companion files in this folder: `greatlakes_setup_env.sh` (build the env once),
`greatlakes_train.sbatch` (unattended batch run), `run_train.py` (the launcher).

## Variables used below

| name | value (adjust if you named things differently) |
|---|---|
| uniqname | `yhling` |
| account | `mdraelos0` |
| partition | `spgpu` (A40, 48 GB — biggest+cheapest of the GPU partitions) |
| personal turbo folder (holds conda + projects) | `/nfs/turbo/coe-igmr-pub/yhling` |
| conda env | `/nfs/turbo/coe-igmr-pub/yhling/miniconda3` |
| repo on turbo | `/nfs/turbo/coe-igmr-pub/yhling/multi-view` |
| local mount point | `~/gl-turbo` |

## Step 1 — seed the repo onto turbo (one time, from THIS laptop)

The training run needs only the **code** plus **one** data folder —
`data/facescape/HRNet_train/` (1.7 G). The rest of the tree is dead weight for
this job: `data/` is ~18 G (8.7 G `virtual_camera_data`, a redundant 1.6 G
`.tar.gz`, raw subject dumps) and `output/` is ~14 G of old results. Both venvs
(~7 G) are laptop-only. So exclude all of `data/`/`output/` and send the bundle
separately — ~1.85 G total instead of 30 G+.

First create the destination (rsync 3.1.x won't make parent dirs, and your turbo
folder is new), once, on a login node:

```bash
mkdir -p /nfs/turbo/coe-igmr-pub/yhling/multi-view
```

Use the dedicated transfer node `greatlakes-xfer.arc-ts.umich.edu` (ARC's
recommended host for big copies) rather than a login node.

**1a — code only** (~130 M):

```bash
rsync -avhP \
  --exclude '.venv' --exclude '.venv_facescape' --exclude '__pycache__' \
  --exclude 'output' --exclude 'scratch' --exclude 'data' \
  /home/carson/Documents/Work/IGMR/multi-view/ \
  yhling@greatlakes-xfer.arc-ts.umich.edu:/nfs/turbo/coe-igmr-pub/yhling/multi-view/
```

**1b — the training bundle** (1.7 G):

```bash
rsync -avhP \
  /home/carson/Documents/Work/IGMR/multi-view/data/facescape/HRNet_train/ \
  yhling@greatlakes-xfer.arc-ts.umich.edu:/nfs/turbo/coe-igmr-pub/yhling/multi-view/data/facescape/HRNet_train/
```

The eval sets (`data/WFLW`, `data/AFLW2000`) are intentionally NOT sent — Step 5
runs the real-image eval back on the laptop. Send them the same way only if you
later decide to eval on the cluster. rsync is incremental and `-P`-resumable, so
re-running either command after an interrupt picks up where it left off.

## Step 2 — mount turbo locally (for editing code)

```bash
mkdir -p ~/gl-turbo
sshfs greatlakes:/nfs/turbo/coe-igmr-pub/yhling ~/gl-turbo
# unmount later with:  fusermount -u ~/gl-turbo
```

Edit code under `~/gl-turbo/multi-view/...` (NOT in VS Code's "remote" sidebar
window). Saves land on turbo instantly and the compute node sees them.

## Step 3 — build the env (one time, on a LOGIN node)

```bash
ssh greatlakes
bash /nfs/turbo/coe-igmr-pub/yhling/multi-view/scripts/facescape/hrnet/greatlakes_setup_env.sh
```

Installs miniconda + a `hrnet` env on turbo and the deps
(`torch torchvision tensorboardX opencv-python numpy pandas Pillow yacs hdf5storage`
+ scipy/matplotlib for later eval). `cuda avail: False` here is expected — login
nodes have no GPU.

## Step 4 — interactive training run (preferred for the first run)

```bash
ssh greatlakes                 # note the login node, e.g. gl-login2
tmux new -s hrnet              # remember the session name "hrnet"

# request one A40 for up to 6h (job vanishes — and billing stops — when it ends early)
salloc --partition spgpu --account mdraelos0 --nodes 1 \
       --cpus-per-gpu 8 --gpus 1 --mem-per-gpu=48G --time 06:00:00

# once the prompt lands on a compute node:
cd /nfs/turbo/coe-igmr-pub/yhling/multi-view
source /nfs/turbo/coe-igmr-pub/yhling/miniconda3/etc/profile.d/conda.sh
conda activate hrnet
python -c "import torch; print('cuda', torch.cuda.is_available())"   # expect True now

python scripts/facescape/hrnet/run_train.py \
  --cfg scripts/facescape/hrnet/face_alignment_facescape_w18_scratch.yaml
```

What to watch (per the forte log analysis): synth-val NME bounces 0.7–3.0 for
~10 epochs, snaps to ~0.07, then declines to ~0.039 and goes flat by epoch ~42.
TRAIN NME staying ~0.22 is an aug artifact, not underfitting — read the TEST curve.
If val NME does NOT approach ~0.039, the aug made the synthetic task harder
(somewhat expected) rather than a config bug.

### detach / reconnect / clean up

- Detach tmux without stopping the job: `Ctrl-b` then `d`.
- Reconnect: `ssh greatlakes` -> `ssh gl-loginX` (same login node) -> `tmux a -t hrnet`.
- Lost the allocation shell but job still alive: `srun --jobid=XXXX --pty bash`.
- Check / free resources: `squeue -u yhling`  then  `scancel JOBID` when done.
  Always scancel once training finishes so you stop holding the GPU.

## Step 4-alt — unattended batch run

```bash
cd /nfs/turbo/coe-igmr-pub/yhling/multi-view
sbatch scripts/facescape/hrnet/greatlakes_train.sbatch   # emails on BEGIN/END/FAIL
squeue -u yhling           # monitor
tail -f log/slurm-*.out    # live output
```

## Step 5 — pull the model back and run the real eval

The run writes to `output/300W/face_alignment_facescape_w18_scratch/` on turbo.
It is already visible at `~/gl-turbo/multi-view/output/...` via the mount, or copy
it explicitly:

```bash
rsync -avhP \
  greatlakes:/nfs/turbo/coe-igmr-pub/yhling/multi-view/output/300W/face_alignment_facescape_w18_scratch/ \
  /home/carson/Documents/Work/IGMR/multi-view/output/hrnet/iter1_scratch_aug/
```

Then re-run `eval_real/run_eval.py` on AFLW2000 + WFLW (see `eval_real/README.md`).
New real NME vs the old ~0.49 IS the iteration-1 result.
