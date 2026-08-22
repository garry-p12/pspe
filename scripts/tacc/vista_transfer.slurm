#!/bin/bash
#SBATCH -J pspe-transfer
#SBATCH -o %x.o%A_%a
#SBATCH -e %x.e%A_%a
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --array=0-4
#
# Phase F on Vista (GH200): one seed per array task. Seeds are independent runs,
# so the sweep costs one seed's wall-clock instead of five. Each task writes its
# own runs/seeds_full/seed_<N>/results.json — the layout
# `eval/run_seeds.py --aggregate-only` expects, so aggregation happens back on
# the laptop with no cluster-side merge step.
#
#   sbatch -A <allocation> scripts/tacc/vista_seeds.slurm
set -euo pipefail

VENV="${VENV:-$WORK/pspe-venv}"
REPO="${REPO:-$WORK/pspe}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"

module purge
# Hierarchical Lmod: gcc first, or cuda/python3 do not resolve.
module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"
cd "$REPO"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# get_device("cuda") falls back to CPU without complaint, so a broken CUDA build
# would produce slow CPU numbers labelled as GPU numbers. Fail loudly instead.
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA visible to torch'"

# The transfer matrix trains one padded surrogate per family, so all three
# datasets must exist before the array starts — same write-race reason as the
# other sweeps, tripled. Seed 0 is not special here: every task needs all three.
for FAMILY in dar swe rdf; do
    DATA="$REPO/data/pde_testbeds/${FAMILY}_64.npz"
    if [ ! -f "$DATA" ]; then
        echo "missing $DATA — generate or rsync all three families before submitting" >&2
        exit 1
    fi
done

# The full 3x3 cross-family matrix at reportable scale. The README's current
# matrix is one seed at 32^2 / 8 epochs and says so; this is 64^2 / 20 epochs
# across five seeds.
srun python eval/run_transfer.py \
    --seed "$SEED" \
    --grid 64 \
    --epochs 20 \
    --full \
    --device cuda \
    --out "runs/transfer_seeds/seed_${SEED}"
