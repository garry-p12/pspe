#!/bin/bash
#SBATCH -J pspe-fire20
#SBATCH -o %x.o%A_%a
#SBATCH -e %x.e%A_%a
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH --array=0-19
#
# Wildfire planning at 20 seeds. The +12.0 point gap over operational practice
# rests on 5 seeds (t = 11.15, df 4) with two of the five table rows carried at
# 3 seeds; it is the project's only real-data decision result and the thinnest
# headline.
#
# ONE SEED PER TASK, deliberately. The earlier pspe-tplan runs took 33 minutes
# per seed at one seed per node. Packing five onto a single GH200 would put four
# sequential batches past two hours before GPU contention, and the first version
# of this job did exactly that and would have hit its wall clock. A seed that
# overruns should cost that seed, not the experiment -- and twenty one-hour
# tasks backfill on a queue 238 deep where two three-hour tasks wait.
#
# The 1,500 held-out fires are a FIXED archive: the seed varies initialisation
# over a fixed test set rather than resampling evaluation conditions, so defect
# 46 does not apply and more seeds simply narrow the interval.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"; SEED="${SLURM_ARRAY_TASK_ID:-0}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"
OUT=runs/ndws_planning_20
mkdir -p "$OUT"
srun python -u eval/run_ndws_planning.py --seed "$SEED" --horizon 3 --device cuda \
    --out "$OUT/seed_${SEED}"
