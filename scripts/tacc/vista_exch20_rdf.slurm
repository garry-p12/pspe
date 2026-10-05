#!/bin/bash
#SBATCH -J pspe-exchrdf
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#
# Exchangeability at 20 seeds, ON RDF.
#
# The companion job ran 20 seeds without --testbed, which defaults to dar --
# the family where exchangeability HELD. It came back 0/20 rejections on all
# three tests, which is a real result and not the one that was needed: the
# published claim is that exchangeability FAILS on one of two testbeds, and
# that failure is on rdf, still at 3 seeds (p = 0.000/0.002/0.000).
#
# A rejection established on 3 seeds is the weakest link in the guarantee's
# own assumption, and it is the direction a reviewer will press, because
# "assumption violated, coverage held anyway" is a claim about robustness that
# needs the violation itself to be solid. The assumption the entire guarantee rests on is
# currently tested on 3 seeds and 320 scores, and the test REJECTS on one of two
# testbeds. A load-bearing assumption checked at n = 3 is what a reviewer asks
# about first.
#
# PACKED into one task on purpose, where the wildfire job is one seed per task.
# The two are sized from their own measured cost rather than from a convention:
# an exchangeability seed takes 5:20, so twenty of them in batches of five is
# ~27 minutes and fits one slot comfortably, while a wildfire seed takes 33
# minutes and five on one GH200 would put four batches past any sensible wall
# clock. Packing also matters because the QOS allows 40 submitted tasks per
# user and the wildfire array already holds twenty.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"
OUT=runs/exchangeability_20_rdf
mkdir -p "$OUT"
for BATCH in "0 1 2 3 4" "5 6 7 8 9" "10 11 12 13 14" "15 16 17 18 19"; do
  for s in $BATCH; do
    python -u eval/run_exchangeability.py --seeds "$s" --testbed rdf --grid 64 --epochs 20 \
      --iterations 400 --probe-every 10 --delta 0.1 --calib-frac 0.5 \
      --n-perm 2000 --device cuda --out "$OUT/seed_$s" > "$OUT/seed_$s.log" 2>&1 &
  done
  wait
  echo "=== exchangeability batch $BATCH done ===" >&2
done
echo "=== complete: $(ls -d $OUT/seed_* 2>/dev/null | wc -l) seeds ===" >&2
