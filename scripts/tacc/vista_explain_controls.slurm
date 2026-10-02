#!/bin/bash
#SBATCH -J pspe-expl-ctl
#SBATCH -o %x.o%A_%a
#SBATCH -e %x.e%A_%a
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH --array=0-2
#
# The Explain arms, rerun for two reasons that each invalidate the old numbers.
#
# 1. max_len was 96 against reference briefs averaging 136 tokens, so every
#    generated brief was cut before the cost/reward/confidence footer and the
#    parsed std fell back to 0.4. The F ~ 0.57 plateau was a truncation
#    ceiling, not a learning limit (defect 47). Now 160.
#
# 2. The post-hoc control was an untrained generator -- posthoc=True sets
#    iterations to 0 -- so "trained-in beats post-hoc" was trained vs
#    untrained (defect 48). Two real controls now run beside it:
#      post-hoc-transfer  trained on a DIFFERENT policy's briefs, then asked
#                         to explain this one with no further training
#      post-hoc-prompted  the instruction-tuned backbone prompted in its OWN
#                         tokens, no training, no prefix, no LoRA
#
# The prompted arm is the only path on which the LM is used as a language
# model: everywhere else its input embedding and output head are both bypassed
# (defect 49), so it has never been asked to write anything.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"; SEED="${SLURM_ARRAY_TASK_ID:-0}"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export HF_HUB_OFFLINE=1
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"

# Reuse the planner these results were computed against, so the only things
# that change are max_len and the arm set. Train only if it is missing.
CKPT="runs/explain_real_v2/seed_${SEED}/policy.pt"
if [[ ! -f "$CKPT" ]]; then
  echo "no checkpoint at $CKPT, training one"
  srun python scripts/train_plan.py seed="$SEED" device=cuda planner.iterations=200 \
      output.checkpoint="$CKPT" hydra.run.dir="outputs/explain_ctl_plan_${SEED}"
else
  echo "reusing $CKPT"
fi

# Fail fast. Two of the five arms have never run on a real backbone, and the
# prompted one touches a code path (native tokenizer, chat template, generate)
# that nothing else in this repo uses. A two-minute pass over all five arms at
# grid 16 / 2 iterations costs almost nothing and turns a four-hour silent
# failure into an immediate one; `set -e` aborts the job if it trips.
echo "=== smoke: all five arms, tiny ==="
srun python eval/run_explain_baselines.py \
    --testbed dar --grid 16 --iterations 2 --batch 2 --n-permutation 4 \
    --device cuda --backbone Qwen/Qwen2.5-0.5B-Instruct --max-len 160 \
    --policy "$CKPT" --seed "$SEED" \
    --out "runs/explain_controls/smoke_${SEED}"
echo "=== smoke passed, real run ==="

srun python eval/run_explain_baselines.py \
    --testbed dar --grid 64 --iterations 200 --device cuda \
    --backbone Qwen/Qwen2.5-0.5B-Instruct --max-len 160 \
    --policy "$CKPT" --seed "$SEED" \
    --out "runs/explain_controls/seed_${SEED}"
