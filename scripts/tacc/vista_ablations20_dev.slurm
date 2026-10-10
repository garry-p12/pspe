#!/bin/bash
#SBATCH -J pspe-abl-dev
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#
# A HEDGE for the ablation rerun, on the development partition.
#
# The gh array (1052011) has been pending with no estimated start while gh sits
# at 342 pending against 141 running; gh-dev is 28 against 9. Prior ablation
# seeds took 11:06-11:53 each, so twenty of them in four batches of five is
# well inside two hours in a single task -- and one packed task is what the dev
# QOS allows (MaxJobsPU = 1).
#
# Writes to its own directory so the two submissions cannot interleave partial
# writes into the same seed. Whichever starts first is the one we use; the other
# gets cancelled rather than left to burn an allocation.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"
OUT=runs/ablations_20_dev
mkdir -p "$OUT"
for BATCH in "0 1 2 3 4" "5 6 7 8 9" "10 11 12 13 14" "15 16 17 18 19"; do
  for s in $BATCH; do
    python -u eval/run_ablations.py --testbed dar --grid 64 --seed "$s" \
      --device cuda --out "$OUT/seed_$s" > "$OUT/seed_$s.log" 2>&1 &
  done
  wait
  echo "=== batch $BATCH done ===" >&2
done
echo "=== complete: $(ls -d $OUT/seed_* 2>/dev/null | wc -l) seeds ===" >&2
