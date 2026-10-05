#!/bin/bash
#SBATCH -J pspe-rho
#SBATCH -o %x.o%A_%a
#SBATCH -e %x.e%A_%a
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --array=0-1
#
# Does Proposition 3's closed form predict a LEARNED planner's violation rate?
#
# R(rho) = 1 - Phi(z_delta / rho) is validated to 0.003 in runs/margin_synthetic/,
# where sigma and epsilon are dialled by hand and nothing is learned. It has
# never been tested on a real planner, because sigma_ep and epsilon were
# computed at every probe and discarded. They are now summarised, and this run
# puts them beside the realised violation rate of the recipe the formula is
# about -- the model-error margin, which Proposition 3 excludes.
#
# Five arms per seed: baseline, probe, margin, probe+margin (ours), and
# probe+model-error (the excluded recipe). 20 seeds per testbed.
#
# PRE-REGISTERED, before the first run:
#   - rho > 1 on both families, because the realised-cost spread exceeds the
#     matched-pair model error wherever the dual is active (section 9 measured
#     rho from 1.20 to 96.24 on rdf by a different route);
#   - the model-error arm's realised violation should sit NEAR R(rho) computed
#     from that seed's own measured rho, and ABOVE delta;
#   - our probe+margin arm should sit near delta and well below R(rho), because
#     it conformalises the quantity the dual controls instead.
#
# What would falsify the closed form as a description of the real system: the
# model-error arm's violation rate landing far from R(rho) -- in which case the
# theorem's mechanism holds in the Gaussian setting it was proved for and does
# not transfer, and that is the honest finding. A back-of-envelope at rho = 1.40
# predicts 0.180 against rdf's previously measured 0.182, but that rho was
# chosen to produce the match, so it is a hypothesis and not a result.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) TB=rdf
     EXTRA="--probe-every 10 --sat-coef 1.0 --adv-floor 1e-2 --dual-ema 0.7 --ki 0.5 \
            --margin-conformal --margin-delta 0.1" ;;
  1) TB=dar
     EXTRA="--probe-every 20 --margin-k 2.0 --margin-conformal --margin-delta 0.1" ;;
  *) echo "unexpected array index" >&2; exit 1 ;;
esac
[ -f "data/pde_testbeds/${TB}_64.npz" ] || { echo "missing ${TB}_64.npz" >&2; exit 1; }

OUT="runs/rho_test_${TB}"
mkdir -p "$OUT"
echo "=== $TB, 20 seeds, rho measured + model-error arm ===" >&2
for BATCH in "0 1 2 3 4" "5 6 7 8 9" "10 11 12 13 14" "15 16 17 18 19"; do
  for s in $BATCH; do
    python -u eval/run_constraint_fix.py --testbed "$TB" --seed "$s" --grid 64 \
      --epochs 20 --iterations 200 $EXTRA \
      --device cuda --out "$OUT/seed_$s" > "$OUT/seed_$s.log" 2>&1 &
  done
  wait
  echo "=== $TB batch $BATCH done ===" >&2
done
echo "=== $TB complete: $(ls -d $OUT/seed_* 2>/dev/null | wc -l) seeds ===" >&2
