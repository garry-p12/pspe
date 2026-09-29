#!/bin/bash
#SBATCH -J t3-flooddelta
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#
# T3.1 (section 8.5): the delta-sweep on a REAL HAZARD.
#
# The reviewer's sharpest structural point: "the strongest real-world-looking
# experiment validates the planner, not the proposed safety method." The
# strongest evidence for the method is the delta-sweep -- the default recipe's
# realised violation rate goes insensitive to what you request (0.30 realised
# when 0.02 is asked) while the residual recipe tracks it -- and that figure
# exists only on the PDE testbeds.
#
# Here it runs with the LISFLOOD-FP local-inertial solver as reality, on the
# real levee-design problem: heights at K sites under a construction budget,
# hydrograph magnitude unknown at build time, limit declared on realised depth.
#
# SIGMA CHOICE IS THE WHOLE EXPERIMENT. Proposition 2's acceptance test puts the
# feasibility crossing between q-log-sigma 0.08 and 0.12 (section 5.6c). Above
# it the required margin exceeds the headroom, every arm is degenerate, and
# coverage is an artefact of infeasibility rather than of the recipe -- the
# script detects and labels that. So the sweep runs BELOW the crossing, where
# the comparison is meaningful, and one arm above it to show the acceptance test
# predicting its own failure.
#
# gg rather than gh: the solver is CPU-friendly and gh is at 564/576 with 135
# queued, while gg has ~89 idle nodes.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
module purge && module load gcc/13.2.0 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"

# The limit must sit where q < limit AND violations still occur. The smoke runs
# bracket it: at limit 0.02 the delta=0.1 margin (0.041) is degenerate, at 0.06
# nothing violates. So scan, rather than assume -- and the scan is cheap because
# limits and deltas share one surrogate and one calibration set.
LIMITS="0.02 0.03 0.05 0.08"
DELTAS="0.02 0.05 0.10 0.20 0.30"
for SIG in 0.04 0.20; do
  TAG=$(echo "$SIG" | tr -d '.')
  echo "=== q-log-sigma $SIG ===" >&2
  python -u eval/run_flood_margin.py \
      --q-log-sigma "$SIG" --limits $LIMITS --deltas $DELTAS \
      --arms none model_error residual \
      --n-train 600 --n-cal 200 --n-eval 200 --epochs 300 \
      --device cpu --out "runs/flood_delta/sig${TAG}" \
      > "runs/flood_delta_sig${TAG}.log" 2>&1
done

echo "=== sweep tables ===" >&2
for d in runs/flood_delta/sig*/; do echo "--- $d"; cat "$d/sweep.md" 2>/dev/null; done
