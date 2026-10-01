#!/bin/bash
#SBATCH -J t21-exact-cpu
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#
# T2.1 exact delta, CPU route on gg.
#
# WHY A SECOND ROUTE RATHER THAN ANOTHER RESUBMISSION: PriorityWeightAge is
# 7000 -- the largest term by far, accruing over 14 days -- and every cancel and
# resubmit resets it to zero. Today's moves (gh -> gb -> gh) threw away that
# priority three times, which is why these jobs sit at position 245 of 352 with
# an AGE component of 18. The gh jobs are now left alone to age. This is an
# ADDITIONAL job on an uncontended partition, not a replacement.
#
# gg is CPU. The earlier CPU run did 2 seeds in 3:51 (t0-exch12), and one seed
# alone timed out at 4 h (t0-exch), so 3 seeds in parallel with a 6 h wall is
# the right shape. Only rdf: it is the family that fails the lag-1 test and so
# the one whose exact drift term matters.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
module purge && module load gcc/13.2.0 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"

for SEED in 0 1 2; do
  python -u eval/run_exchangeability.py --testbed rdf --seeds "$SEED" \
      --grid 64 --epochs 20 --iterations 400 --probe-every 10 \
      --delta 0.1 --calib-frac 0.5 --n-perm 2000 --device cpu \
      --out "runs/exch_exact_rdf/seed_${SEED}" \
      > "runs/exch_exact_rdf_seed_${SEED}.log" 2>&1 &
done
wait
python eval/run_exchangeability.py --aggregate --out runs/exch_exact_rdf || true
echo "=== gg exact-delta done ===" >&2
