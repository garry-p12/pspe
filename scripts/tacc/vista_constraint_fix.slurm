#!/bin/bash
#SBATCH -J pspe-cfix
#SBATCH -o %x.o%A_%a
#SBATCH -e %x.e%A_%a
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --array=0-4
#
# Does closing the surrogate/reality cost gap fix the constraint violation?
# Four arms per seed (baseline / probe / margin / probe+margin), one seed per
# node. The planner exceeded the limit on 7.3% of evaluations against 0% for
# every model-free baseline; this is the experiment that says whether the fix
# works and what it costs in real samples and return.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"; SEED="${SLURM_ARRAY_TASK_ID:-0}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"
[ -f data/pde_testbeds/dar_64.npz ] || { echo "missing dar_64.npz" >&2; exit 1; }
srun python eval/run_constraint_fix.py --seed "$SEED" --grid 64 --epochs 20 \
    --iterations 200 --probe-every 20 --margin-k 2.0 --device cuda \
    --out "runs/constraint_fix/seed_${SEED}"
