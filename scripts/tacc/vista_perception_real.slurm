#!/bin/bash
#SBATCH -J pspe-perc-real
#SBATCH -o %x.o%A_%a
#SBATCH -e %x.e%A_%a
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH --array=0-2
#
# Phase C: the perception arms on a REAL open-weight vision backbone (SigLIP),
# not the in-tree stub. Every perception number in the repo so far carries
# backbone_is_stub: true; these are the first that will not.
#
# HF_HOME points at the login-node-populated cache: compute nodes have no
# outbound network, so an un-cached model id would fail here, not download.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"; SEED="${SLURM_ARRAY_TASK_ID:-0}"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export HF_HUB_OFFLINE=1
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"
srun python eval/run_perception_baselines.py \
    --testbed dar --grid 64 --epochs 20 --device cuda \
    --backbone google/siglip-base-patch16-224 \
    --seed "$SEED" --out "runs/perception_real/seed_${SEED}"
