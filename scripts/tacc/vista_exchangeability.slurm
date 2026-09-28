#!/bin/bash
#SBATCH -J pspe-exch
#SBATCH -o %x.o%A_%a
#SBATCH -e %x.e%A_%a
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --array=0-2
#
# Are the conformal margin's calibration scores exchangeable across a run?
#
# Section 3.2 calibrates the margin on residuals collected DURING training, and
# a learning policy is a distribution shift, so split conformal's assumption is
# not free. Every coverage result we have draws its scores i.i.d., which tests
# the estimator and not the assumption (section 9.3). One seed per node: a seed
# that overruns costs that seed rather than the experiment.
#
# Sizing: run_constraint_fix fits 4 arms x 200 iterations inside 2 h on this
# partition. This is 1 arm x 400 iterations, so roughly half that work, and the
# same budget is deliberately generous.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"; SEED="${SLURM_ARRAY_TASK_ID:-0}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"
[ -f data/pde_testbeds/dar_64.npz ] || { echo "missing dar_64.npz" >&2; exit 1; }

# 40 probes x 8 episodes = 320 scores; a 50/50 split leaves the quantile at
# rank 145 of 160, which is an order statistic rather than the sample maximum,
# and resolves delta = 0.1 to 1/160. Fewer probes than this cannot (defect 18).
srun python eval/run_exchangeability.py --seeds "$SEED" --grid 64 --epochs 20 \
    --iterations 400 --probe-every 10 --delta 0.1 --calib-frac 0.5 \
    --n-perm 2000 --device cuda --out "runs/exchangeability/seed_${SEED}"
