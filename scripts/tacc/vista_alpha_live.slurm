#!/bin/bash
#SBATCH -J pspe-alpha-live
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#
# Does the mixing rule do anything on the flood twin? (baselines/run_alpha_live.py)
#
#   sbatch -A ATM23014 --export=ALL scripts/tacc/vista_alpha_live.slurm
#   sbatch -A ATM23014 --export=ALL,SEEDS="0 1 2 3 4 5 6 7 8 9 10 11" ... 
#
# Every flood result in this project so far ran with alpha pinned to zero,
# because FloodOperateEnv defaults to differentiable=False and the trainer
# then sets alpha_fixed=0. This runs the arm where alpha can actually move,
# against a matched control that differs ONLY in that.
#
# CPU, not GPU, and deliberately. FloodSolver at grid 32 is small; the win
# here is 24 independent runs across the node's cores, not one run on the
# GH200. THREADS x concurrent runs is sized to stay under the core count.
set -uo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"

OUT="${OUT:-runs/alpha_live}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7}"
ARMS="${ARMS:-adaptive alpha0 truth}"
# v1 is the defaults every flood result in this project has used. v2 turns on
# the saturation wall and the normalised dual that PlannerConfig's own notes
# prescribe for exactly the failure the polder exhibits -- pre-tanh past 3 by
# iteration 30 and reaching 1,436, alpha 0.979 on a dead branch, lambda 27.7
# climbing linearly and binding nothing. v1 is kept as the control, because a
# fix with no documented "before" is an assertion.
CONFIGS="${CONFIGS:-v1 v2}"
# 3 threads x 24 concurrent fits the node and gets the 48 runs into two waves.
THREADS="${THREADS:-3}"
ITERS="${ITERS:-150}"

mkdir -p "$OUT"

# The surrogate is fitted ONCE, before any arm starts, and shared. Fitting it
# per arm would make the comparison "two draws of a network" rather than
# "alpha on versus alpha off", which is the §4.2a lesson about matched arms.
python -m baselines.run_alpha_live --fit --out "$OUT" \
  --iterations "$ITERS" --threads "$THREADS" || exit 1

# Collect PIDs as launched. Do NOT `wait` inside the loop: that reaps the
# jobs, leaving the exit-code loop below nothing to read, which has silently
# turned failed runs into rc=0 on this cluster before.
PIDS=()
for cfg in $CONFIGS; do
  for arm in $ARMS; do
    for s in $SEEDS; do
      python -m baselines.run_alpha_live --arm "$arm" --config "$cfg" \
        --seed "$s" --out "$OUT" --iterations "$ITERS" --threads "$THREADS" &
      PIDS+=($!)
    done
  done
done

rc=0
for p in "${PIDS[@]}"; do wait "$p" || rc=$?; done

python -m baselines.run_alpha_live --aggregate --out "$OUT"
echo "alpha_live done rc=$rc"
exit "$rc"
