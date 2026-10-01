#!/bin/bash
#SBATCH -J t3-floodbudget
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 10:00:00
#
# T3.1, third attempt. The first two failed for opposite reasons and both are
# recorded because the pattern is the point.
#
#   attempt 1  single-start planner stuck 30x off the achievable frontier
#              (0.159 m realised against 0.005 m achievable). Every arm violated
#              every episode -> no recipe distinguishable.
#   attempt 2  multi-start fixed the planner, and the constraint went slack the
#              other way. At limits <= 0.05 the margin exceeds the limit and every
#              arm is degenerate; at 0.08 the margin fits but the planner meets
#              the tightened target trivially and violations collapse to 0.005 -
#              0.035 against requested deltas of 0.02 - 0.30. Conservative
#              everywhere, so still no separation.
#
# The diagnosis is BUDGET, not limit. Achievable depth spans 0.005 - 0.267 m, so
# a budget of 1.5 lets the planner hit any target it is given and the constraint
# never bites. The delta-sweep can only be informative where meeting `limit - q`
# actually costs something.
#
# So scan the budget down until the achievable optimum sits NEAR the limit. The
# marker to look for in the output: the no-margin arm violating well above delta
# while the margined arms land near it, rather than both at zero.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
module purge && module load gcc/13.2.0 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"

for BUD in 0.8 0.5 0.3; do
  TAG=$(echo "$BUD" | tr -d '.')
  echo "=== budget $BUD ===" >&2
  python -u eval/run_flood_margin.py \
      --q-log-sigma 0.04 --budget "$BUD" \
      --limits 0.03 0.05 0.08 --deltas 0.05 0.10 0.20 0.30 \
      --arms none model_error residual \
      --n-train 600 --n-cal 200 --n-eval 200 --epochs 300 \
      --device cpu --out "runs/flood_budget/b${TAG}" \
      > "runs/flood_budget_b${TAG}.log" 2>&1
  echo "--- budget $BUD diagnostics ---" >&2
  grep -E "span scan|realised depth mean|sigma =|verdict" "runs/flood_budget_b${TAG}.log" >&2 || true
done

echo "=== sweep tables ===" >&2
for d in runs/flood_budget/b*/; do echo "--- $d"; cat "$d/sweep.md" 2>/dev/null; done
