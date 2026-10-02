#!/bin/bash
#SBATCH -J a1-swe-m
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:15:00
#
# Audit A1, stage 2: the margin comparison on revived swe, 5 seeds per job.
# Submitted as short jobs rather than one long one, because gh backfills short
# work into gaps -- the whole point of splitting. SEEDS is passed in.
#
# The walltime has to match that intent. It was 02:30:00 against a measured
# 57:32-58:13 for five parallel seeds, which is 2.6x the runtime: a request
# that large cannot be backfilled into a one-hour gap, so the splitting bought
# nothing and twelve jobs sat PENDING on a fully allocated partition behind a
# limit they did not need. 01:15:00 is 29% headroom over the slowest observed
# seed. Splitting a job for backfill and then asking for more wall time than it
# uses is the same mistake as bounding one resource and assuming the other
# follows (section 5.14's dx floor).
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
LIMIT=$(cat runs/swe_anchor/limit.txt)
echo "using limit $LIMIT for seeds ${SEEDS}" >&2
for s in ${SEEDS}; do
  python -u eval/run_constraint_fix.py --testbed swe --grid 64 --epochs 20 \
    --horizon 96 --task-target 0.12 --task-umax 0.14 --task-budget 0.60 \
    --sat-coef 1.0 --adv-floor 1e-2 --dual-ema 0.7 --ki 0.5 --device cuda \
    --seed $s --iterations 200 --probe-every 10 --task-cost-limit "$LIMIT" \
    --margin-conformal --margin-mode residual --margin-delta 0.1 \
    --out "runs/constraint_fix_swe20/seed_$s" > "runs/constraint_fix_swe20_$s.log" 2>&1 &
done
wait
