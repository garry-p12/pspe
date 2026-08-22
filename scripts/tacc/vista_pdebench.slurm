#!/bin/bash
#SBATCH -J pspe-pdeb
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#
# Phase A: the surrogates on a real community benchmark, against PDEBench's
# published FNO error. Every surrogate number in this repo so far has been
# scored against this repo's own solver, which cannot be wrong in the same
# direction as the model and so cannot detect a shared bias.
#
#   sbatch -A <allocation> scripts/tacc/vista_pdebench.slurm
set -euo pipefail

VENV="${VENV:-$WORK/pspe-venv}"
REPO="${REPO:-$WORK/pspe}"

module purge
module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"
cd "$REPO"

python -c "import torch; assert torch.cuda.is_available(), 'no CUDA visible to torch'"

FILE="data/pdebench/2D_rdb_NA_NA.h5"
[ -f "$FILE" ] || { echo "missing $FILE — run scripts/tacc/download_pdebench.sh first" >&2; exit 1; }

srun python eval/run_pdebench.py \
    --file "$FILE" --testbed swe --benchmark 2d_shallowwater \
    --grid 64 --epochs 20 --n-samples 512 --device cuda \
    --operators fno deeponet gnot
