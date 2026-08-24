#!/bin/bash
#SBATCH -J pspe-res
#SBATCH -o %x.o%A_%a
#SBATCH -e %x.e%A_%a
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --array=0-4
#
# Section 7.4 resolution generalization, across seeds and up to 128^2 — the
# "tens of thousands of grid cells" the proposal claims (128^2 = 16,384).
# Trains at 64^2 and evaluates at 64/96/128, so the FNO's discretization
# invariance is tested rather than asserted.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"; SEED="${SLURM_ARRAY_TASK_ID:-0}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"
srun python eval/run_resolution.py --testbed dar --train-grid 64 \
    --eval-grids 64 96 128 --epochs 20 --seed "$SEED" --device cuda \
    --out "runs/resolution_seeds/seed_${SEED}"
