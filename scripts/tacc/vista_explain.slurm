#!/bin/bash
#SBATCH -J pspe-expl
#SBATCH -o %x.o%A_%a
#SBATCH -e %x.e%A_%a
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --array=0-4
#
# Section 7.2 explanation control: trained-in vs no-faithfulness vs post-hoc.
#
# The policy has to be *trained* first. Briefs about an untrained policy are
# degenerate — it actuates almost nothing, so every arm describes the same
# do-nothing behaviour and the three arms cannot separate no matter what the
# faithfulness objective does. So each task trains its own planner, then
# explains it.
#
#   sbatch -A <allocation> scripts/tacc/vista_explain.slurm
set -euo pipefail

VENV="${VENV:-$WORK/pspe-venv}"
REPO="${REPO:-$WORK/pspe}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"

module purge
module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"
cd "$REPO"

python -c "import torch; assert torch.cuda.is_available(), 'no CUDA visible to torch'"

CKPT="runs/explain_seeds/seed_${SEED}/policy.pt"
srun python scripts/train_plan.py seed="$SEED" device=cuda \
    planner.iterations=200 \
    output.checkpoint="$CKPT" \
    hydra.run.dir="outputs/explain_plan_${SEED}"

srun python eval/run_explain_baselines.py \
    --testbed dar --grid 64 --iterations 200 --device cuda \
    --policy "$CKPT" --seed "$SEED" --out "runs/explain_seeds/seed_${SEED}"
