#!/bin/bash
#SBATCH -J pspe-expl-v2
#SBATCH -o %x.o%A_%a
#SBATCH -e %x.e%A_%a
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH --array=0-2
#
# Phase D: the explanation arms on a REAL LM (Qwen2.5-0.5B-Instruct) against a
# trained planner. The stub-LM sweep showed the faithfulness term doing nothing
# measurable; that claim is only testable on a backbone that can actually write
# a brief.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"; SEED="${SLURM_ARRAY_TASK_ID:-0}"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export HF_HUB_OFFLINE=1
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"

CKPT="runs/explain_real_v2/seed_${SEED}/policy.pt"
srun python scripts/train_plan.py seed="$SEED" device=cuda planner.iterations=200 \
    output.checkpoint="$CKPT" hydra.run.dir="outputs/explain_real_plan_${SEED}"
srun python eval/run_explain_baselines.py \
    --testbed dar --grid 64 --iterations 200 --device cuda \
    --backbone Qwen/Qwen2.5-0.5B-Instruct \
    --policy "$CKPT" --seed "$SEED" --out "runs/explain_real_v2/seed_${SEED}"
