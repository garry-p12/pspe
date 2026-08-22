#!/bin/bash
#SBATCH -J pspe-perc
#SBATCH -o %x.o%A_%a
#SBATCH -e %x.e%A_%a
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH --array=0-4
#
# Section 7.2 perception controls across seeds: LoRA vs frozen linear probe vs
# from-scratch CNN. One seed per array task, aggregated on the laptop.
#
#   sbatch -A <allocation> scripts/tacc/vista_perception.slurm
set -euo pipefail

VENV="${VENV:-$WORK/pspe-venv}"
REPO="${REPO:-$WORK/pspe}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"

module purge
module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"
cd "$REPO"

python -c "import torch; assert torch.cuda.is_available(), 'no CUDA visible to torch'"

srun python eval/run_perception_baselines.py \
    --testbed dar --grid 64 --epochs 20 --device cuda \
    --seed "$SEED" --out "runs/perception_seeds/seed_${SEED}"
