#!/bin/bash
#SBATCH -J t0-alpha20
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#
# Tier 0b: twenty seeds for a comparative claim that five cannot separate.
# Section 4.2 mixing rule: variance rule vs fixed, CI [-12.1, +6.1] at five seeds.
#
# Split out of the original 8 h combined job: gh backfills short jobs into gaps,
# and an 8 h request rarely fits one. The 20-seed conformal run (job 1032033)
# took 51 minutes, so 2 h is the right size for this shape of work.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"
mkdir -p runs/alpha_rule_20
for BATCH in "0 1 2 3 4" "5 6 7 8 9" "10 11 12 13 14" "15 16 17 18 19"; do
  for s in $BATCH; do
    python -u eval/run_alpha_rule.py --seed "$s" --grid 64 --epochs 20 \
      --iterations 200 --device cuda --out "runs/alpha_rule_20/seed_$s" \
      > "runs/alpha_rule_20_seed_$s.log" 2>&1 &
  done
  wait
  echo "=== batch $BATCH done ===" >&2
done
