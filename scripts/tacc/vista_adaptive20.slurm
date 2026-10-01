#!/bin/bash
#SBATCH -J t2-adaptive20
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#
# T2.2 (section 8.5): adaptive conformal as a first-class arm, 20 seeds on rdf.
#
# Section 9.3a is what makes this necessary rather than optional. The lag-1
# permutation test REJECTS exchangeability on rdf on all three seeds
# (p = 0.000, 0.002, 0.000) -- on the family that carries every margin number in
# section 3.2 -- while the drift tests do not reject, so it is serial dependence
# between consecutive probes, not a shift. Realised coverage was 0.1104 against
# a requested 0.1000, and Gibbs-Candes adaptive conformal landed on 0.1000
# exactly.
#
# That measurement was taken inside run_exchangeability.py, on the calibration
# sequence alone. This asks the question that actually matters: deployed as the
# margin in the planner, at 20 seeds, does adaptive beat split conformal on
# realised violations -- and what does it cost in return?
#
# Same configuration as job 1032033 (section 3.2b), which gave residual
# 14.1% -> 5.0%, paired -9.1 pts, p = 0.0002. Only --margin-mode differs, so the
# two trees are directly comparable seed by seed.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"

OUT=runs/constraint_fix_adaptive_rdf_20
mkdir -p "$OUT"
for BATCH in "0 1 2 3 4" "5 6 7 8 9" "10 11 12 13 14" "15 16 17 18 19"; do
  for s in $BATCH; do
    python -u eval/run_constraint_fix.py --testbed rdf --seed "$s" --grid 64 \
      --epochs 20 --iterations 200 --probe-every 10 \
      --sat-coef 1.0 --adv-floor 1e-2 --dual-ema 0.7 --ki 0.5 \
      --margin-conformal --margin-mode adaptive --margin-delta 0.1 --device cuda \
      --out "$OUT/seed_$s" > "$OUT/seed_$s.log" 2>&1 &
  done
  wait
  echo "=== batch $BATCH done ===" >&2
done

python eval/run_bootstrap_report.py --trees "adaptive20=$OUT" \
    --out runs/bootstrap_report_adaptive20 || true
