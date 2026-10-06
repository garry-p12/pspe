#!/bin/bash
#SBATCH -J pspe-conf-real
#SBATCH -o %x.o%A_%a
#SBATCH -e %x.e%A_%a
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --array=0-2
#
# runs/conformal_real, rerun off the truncation ceiling.
#
# Section 4.4's Explain numbers come from this runner, which took
# ExplainConfig's 96-token default against reference briefs averaging 136
# tokens. Every stored brief in the existing run is exactly 96 tokens, names 7
# of 9 actuators, and never reaches the cost/reward/confidence footer -- so the
# parsed std falls back to 0.4, two actuators default to 0, and the F ~ 0.57
# plateau is the cap rather than a learning limit.
#
# --max-len 160 clears the reference. The controls run already established the
# direction (supervised-only reaches 0.901 against a reference of 0.901, and the
# faithfulness objective costs 0.396), so the purpose here is to make section
# 4.4's conformal floors consistent with that rather than to re-discover it.
#
# Three seeds, matching the existing run's shape so the comparison is like for
# like.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"; SEED="${SLURM_ARRAY_TASK_ID:-0}"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export HF_HUB_OFFLINE=1
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"
OUT=runs/conformal_real_160
mkdir -p "$OUT"
srun python -u eval/run_conformal.py --seed "$SEED" --device cuda \
    --backbone Qwen/Qwen2.5-0.5B-Instruct --max-len 160 \
    --out "$OUT/seed_${SEED}"
