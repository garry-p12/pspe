#!/bin/bash
#SBATCH -J a1-swe
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#
# Audit finding A1 (§8.6): the abstract says "three PDE families" and only rdf
# validates the margin. dar is 0% -> 0% (a regression check) and swe is retired.
# This is the run that earns "two".
#
# §3.4a already revived swe's OBJECTIVE -- horizon 96, target 0.12 gives span
# 47.9% and 37.5% improving directions against rdf's 38%. What is missing is the
# cost limit: §3.4a placed it at ~0.083 using the best of 40 random constant
# actions as the upper anchor, and TASK_SPECS anchors on a TRAINED reward-greedy
# planner, which reaches further. So 0.083 is a lower bound.
#
# Stage 1 measures the trained anchor by running with the constraint switched
# off (a huge limit). Stage 2 places the limit at the 35% point every shipped
# TASK_SPECS entry uses -- dar 35.00%, swe 35.15%, rdf 34.99% -- and runs the
# margin comparison at 20 seeds.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"

TGT=0.12; UMAX=0.14; TBUD=0.60; HOR=96
COMMON="--testbed swe --grid 64 --epochs 20 --horizon $HOR \
  --task-target $TGT --task-umax $UMAX --task-budget $TBUD \
  --sat-coef 1.0 --adv-floor 1e-2 --dual-ema 0.7 --ki 0.5 --device cuda"

echo "=== stage 1: trained reward-greedy anchor (constraint off) ===" >&2
for s in 0 1 2; do
  python -u eval/run_constraint_fix.py $COMMON --seed $s --iterations 200 \
    --probe-every 10 --task-cost-limit 1e6 \
    --out "runs/swe_anchor/seed_$s" > "runs/swe_anchor_$s.log" 2>&1 &
done
wait

LIMIT=$(python - <<'PY'
import json, glob
import numpy as np
c = []
for f in glob.glob("runs/swe_anchor/seed_*/baseline/summary.json"):
    c.append(json.load(open(f))["episode_cost"])
greedy = float(np.mean(c))
dn = 0.0017                      # do-nothing cost at this spec, §3.4a
lim = dn + 0.35 * (greedy - dn)  # the 35% convention every TASK_SPECS entry uses
print(f"{lim:.6f}")
import sys; print(f"anchor greedy cost {greedy:.5f} -> limit {lim:.5f}", file=sys.stderr)
PY
)
echo "=== stage 2: margin comparison at limit $LIMIT, 20 seeds ===" >&2

for BATCH in "0 1 2 3 4" "5 6 7 8 9" "10 11 12 13 14" "15 16 17 18 19"; do
  for s in $BATCH; do
    python -u eval/run_constraint_fix.py $COMMON --seed $s --iterations 200 \
      --probe-every 10 --task-cost-limit "$LIMIT" \
      --margin-conformal --margin-mode residual --margin-delta 0.1 \
      --out "runs/constraint_fix_swe20/seed_$s" > "runs/constraint_fix_swe20_$s.log" 2>&1 &
  done
  wait
done
python eval/run_bootstrap_report.py --trees "swe20=runs/constraint_fix_swe20" \
  --out runs/bootstrap_report_swe20 || true
