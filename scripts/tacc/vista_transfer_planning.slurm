#!/bin/bash
#SBATCH -J pspe-tplan
#SBATCH -o %x.o%A_%a
#SBATCH -e %x.e%A_%a
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH --array=0-4
#
# Section 7.4 planning-transfer gap: how much *reward* a policy gives up when
# its surrogate came from the wrong PDE family, as opposed to how wrong the
# forecasts are. Two planners per off-diagonal pair, so this is the expensive
# transfer variant.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"; SEED="${SLURM_ARRAY_TASK_ID:-0}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"
for FAMILY in dar swe rdf; do
    [ -f "data/pde_testbeds/${FAMILY}_64.npz" ] || { echo "missing ${FAMILY}_64.npz" >&2; exit 1; }
done
srun python eval/run_transfer.py --seed "$SEED" --grid 64 --epochs 20 --full \
    --planning --plan-iterations 200 --device cuda \
    --out "runs/transfer_planning/seed_${SEED}"
