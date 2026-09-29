#!/bin/bash
#SBATCH -J t0-exch12
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 05:00:00
#
# Seeds 1 and 2 of the rdf exchangeability battery, run CONCURRENTLY.
#
# Job 1033237 runs the three seeds sequentially, and on CPU one seed is taking
# 40+ minutes against the ~5 min the dar version took on GPU. Three in series
# would not fit its 4 h walltime. The seeds are independent, so running 1 and 2
# side by side here costs one extra node and removes the timeout risk; 1033237
# still produces seed 0 first, and its later seeds are deterministic repeats of
# these.
#
# The script is silent until a seed finishes (its first print is after the whole
# run), so an empty log is not a stalled job -- worth knowing before killing one.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
module purge && module load gcc/13.2.0 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"

for SEED in 1 2; do
  python -u eval/run_exchangeability.py --testbed rdf --seeds "$SEED" \
      --grid 64 --epochs 20 --iterations 400 --probe-every 10 \
      --delta 0.1 --calib-frac 0.5 --n-perm 2000 --device cpu \
      --out "runs/exchangeability_rdf/seed_${SEED}" \
      > "runs/exchangeability_rdf_seed_${SEED}.log" 2>&1 &
done
wait
echo "=== seeds 1 and 2 done ===" >&2
