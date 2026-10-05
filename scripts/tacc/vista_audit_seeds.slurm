#!/bin/bash
#SBATCH -J pspe-audit
#SBATCH -o %x.o%A_%a
#SBATCH -e %x.e%A_%a
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH --array=0-1
#
# Two audit gaps that are pure compute: our two thinnest headline experiments.
#
# Task 0 -- WILDFIRE AT 20 SEEDS. The +12.0 point gap over operational practice
# rests on 5 seeds (t = 11.15, df 4) with two of the five table rows carried at
# 3 seeds and marked. It is the project's only real-data decision result and the
# thinnest of the headlines. The 1,500 held-out fires are a FIXED archive, so
# the seed varies initialisation over a fixed test set rather than resampling
# evaluation conditions -- defect 46 does not apply here and more seeds simply
# narrow the interval.
#
# Task 1 -- EXCHANGEABILITY AT 20 SEEDS. The assumption the whole guarantee
# rests on is currently tested on 3 seeds and 320 scores, and the test REJECTS
# on one of two testbeds. A load-bearing assumption checked at n = 3 is the kind
# of thing a reviewer asks about first, and the direction of the finding
# (rejected, coverage held anyway) is one we want firmly established rather
# than marginal.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    OUT=runs/ndws_planning_20
    mkdir -p "$OUT"
    echo "=== wildfire planning, 20 seeds ===" >&2
    for BATCH in "0 1 2 3 4" "5 6 7 8 9" "10 11 12 13 14" "15 16 17 18 19"; do
      for s in $BATCH; do
        python -u eval/run_ndws_planning.py --seed "$s" --horizon 3 --device cuda \
          --out "$OUT/seed_$s" > "$OUT/seed_$s.log" 2>&1 &
      done
      wait
      echo "=== wildfire batch $BATCH done ===" >&2
    done
    echo "=== wildfire complete: $(ls -d $OUT/seed_* 2>/dev/null | wc -l) seeds ===" >&2
    ;;
  1)
    OUT=runs/exchangeability_20
    mkdir -p "$OUT"
    echo "=== exchangeability, 20 seeds ===" >&2
    # Four at a time: n-perm 2000 permutations per seed is the expensive part.
    for BATCH in "0 1 2 3" "4 5 6 7" "8 9 10 11" "12 13 14 15" "16 17 18 19"; do
      for s in $BATCH; do
        python -u eval/run_exchangeability.py --seeds "$s" --grid 64 --epochs 20 \
          --iterations 400 --probe-every 10 --delta 0.1 --calib-frac 0.5 \
          --n-perm 2000 --device cuda --out "$OUT/seed_$s" \
          > "$OUT/seed_$s.log" 2>&1 &
      done
      wait
      echo "=== exchangeability batch $BATCH done ===" >&2
    done
    echo "=== exchangeability complete: $(ls -d $OUT/seed_* 2>/dev/null | wc -l) seeds ===" >&2
    ;;
  *) echo "unexpected array index" >&2; exit 1 ;;
esac
