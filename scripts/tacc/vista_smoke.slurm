#!/bin/bash
#SBATCH -J pspe-smoke
#SBATCH -o %x.o%A_%a
#SBATCH -e %x.e%A_%a
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:20:00
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

# The dataset must already exist. `ensure_dataset` generates it on a miss, and
# five array tasks missing at once would race on one .npz path — one task would
# read the file while another is still writing it. Staged by rsync from the
# laptop so every seed also trains on byte-identical data to the CPU runs.
DATA="$REPO/data/pde_testbeds/dar_64.npz"
if [ ! -f "$DATA" ]; then
    echo "missing $DATA — rsync it before submitting (see scripts/tacc/README.md)" >&2
    exit 1
fi

# --full: 20 simulate epochs / 200 planner iterations / 200 baseline iterations,
# the budget the README's Phase 2 table was run at.
# Smoke: quick budgets, one ablation group. Proves CUDA is live on a GH200 and
# the whole code path runs there, before the real array spends five nodes.
srun python eval/run_ablations.py \
    --seed 0 --testbed dar --grid 64 --device cuda \
    --only adaptive --out runs/smoke_gh
