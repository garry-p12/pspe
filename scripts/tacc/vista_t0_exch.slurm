#!/bin/bash
#SBATCH -J t0-exch
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#
# Tier 0a: the exchangeability battery on rdf (section 8.5, T0.1).
#
# The reviewer's most specific criticism: the check exists only on dar, while
# rdf carries every margin number in section 3.2. So the assumption is untested
# exactly where it is load-bearing.
#
# On `gg` (Grace, CPU) rather than `gh`: gh is at 564/576 allocated with 135 jobs
# queued and an estimated start 13 hours out, while gg has 89 idle nodes. This
# run is light -- the dar version was ~5 min a seed -- so CPU is a sensible
# trade for starting now. 4 h is generous headroom for the CPU slowdown.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
module purge && module load gcc/13.2.0 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"

for SEED in 0 1 2; do
  python -u eval/run_exchangeability.py --testbed rdf --seeds "$SEED" \
      --grid 64 --epochs 20 --iterations 400 --probe-every 10 \
      --delta 0.1 --calib-frac 0.5 --n-perm 2000 --device cpu \
      --out "runs/exchangeability_rdf/seed_${SEED}" \
      > "runs/exchangeability_rdf_seed_${SEED}.log" 2>&1
  echo "=== seed $SEED done ===" >&2
done
python eval/run_exchangeability.py --aggregate --out runs/exchangeability_rdf || true
