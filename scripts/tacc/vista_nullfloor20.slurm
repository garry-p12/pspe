#!/bin/bash
#SBATCH -J pspe-null20
#SBATCH -o %x.o%A_%a
#SBATCH -e %x.e%A_%a
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH --array=0-3
#
# The corrected baseline table: 20 seeds, the seed actually reaching the
# configs, a do-nothing floor, and the model-free four at OUR sample budget.
#
# Three defects make the current table unquotable as a comparison:
#
#  1. defect 46 -- run_ablations passed --seed only to the global RNG, so all
#     five "seeds" of section 8.3 shared one evaluation IC set. The contrast
#     holds on that draw; the dispersion does not, and the t-values of +3.84 to
#     +17.95 are computed from an understated SE. run_constraint_fix DOES pass
#     the seed, which is why the corrected table is built here.
#
#  2. no null row. The zero policy returns -2.5645 on dar's published IC set at
#     cost 0.2735 against a limit of 0.936, so it never violates. Against that
#     floor Saute buys +0.027, CPO +0.024, primal-dual NPG +0.021 and
#     PPO-Lagrangian +0.007 -- each with 38,400 real transitions -- while
#     probe+margin buys +0.264 with 4,160. Averaged over five IC draws instead
#     of the single shared one, do-nothing returns -2.3822 and beats all four.
#     The sign of "does safe RL beat doing nothing here" depends on the draw, so
#     the row has to be in the table and the seeds have to vary.
#
#  3. unmatched budget. The baselines get 9.2x the planner's real transitions.
#     --baselines caps them at the planner's own count, in the same seed loop.
#
# PRE-REGISTERED, before the first run. The matched-budget baselines will NOT
# violate: an undertrained policy on these testbeds does almost nothing and
# doing nothing is safe -- the smoke test already showed 0% violation at 96
# transitions. So the prediction is that they come in AT or BELOW the null floor
# on return with 0% violation, i.e. safe and not learning, and that probe+margin
# clears the floor by an order of magnitude more than any of them. If a baseline
# clears the null floor materially at 4,160 transitions, the sample-efficiency
# claim is weaker than section 8.3 implies and that is the finding to report.
#
# Task 0 = dar (section 8.3's testbed), task 1 = rdf (the 14.1% -> 5.0% result,
# whose null floor has never been measured). Each uses its own PUBLISHED
# hyperparameters so the new rows sit beside the old ones.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"

# Two budgets, because they answer two different questions and the first
# version of this script only wired one.
#
#  MATCHED (tasks 0-1): the baselines capped at the planner's own real-sample
#  count. Tests the sample-efficiency claim -- what do they achieve with the
#  data we use? Cheap, and the planner arms run alongside.
#
#  FULL (tasks 2-3): the baselines at their own 38,400 transitions, which is
#  what section 8.3 reported, but with 20 seeds that actually vary. This is the
#  task that corrects that table. --baselines-only skips the planner arms, which
#  tasks 0-1 already produce, and keeps the free null row so each table stands
#  alone.
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) TB=dar; MODE=matched ;;
  1) TB=rdf; MODE=matched ;;
  2) TB=dar; MODE=full ;;
  3) TB=rdf; MODE=full ;;
  *) echo "unexpected array index" >&2; exit 1 ;;
esac

case "$TB" in
  dar) EXTRA="--probe-every 20 --margin-k 2.0" ;;
  rdf) EXTRA="--probe-every 10 --sat-coef 1.0 --adv-floor 1e-2 --dual-ema 0.7 --ki 0.5 \
              --margin-conformal --margin-delta 0.1" ;;
esac

case "$MODE" in
  matched) BARGS="--baselines --baseline-batch 16"
           EXTRA="$EXTRA --epochs 20" ;;
  # 200 x 16 x 12 = 38,400, exactly what section 8.3 gave them.
  full)    BARGS="--baselines --baselines-only --baseline-budget 38400 --baseline-batch 16"
           EXTRA="$EXTRA --epochs 1" ;;   # no planner arms, so no surrogate needed
esac
[ -f "data/pde_testbeds/${TB}_64.npz" ] || { echo "missing ${TB}_64.npz" >&2; exit 1; }

OUT="runs/nullfloor20_${TB}_${MODE}"
mkdir -p "$OUT"
echo "=== $TB, $MODE budget, 20 seeds ===" >&2

for BATCH in "0 1 2 3 4" "5 6 7 8 9" "10 11 12 13 14" "15 16 17 18 19"; do
  echo "=== batch $BATCH ===" >&2
  for s in $BATCH; do
    python -u eval/run_constraint_fix.py --testbed "$TB" --seed "$s" --grid 64 \
      --iterations 200 $EXTRA $BARGS \
      --device cuda --out "$OUT/seed_$s" > "$OUT/seed_$s.log" 2>&1 &
  done
  wait
  echo "=== batch $BATCH done ===" >&2
done

echo "=== $TB $MODE complete; $(ls -d $OUT/seed_* 2>/dev/null | wc -l) seeds ===" >&2
