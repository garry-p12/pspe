#!/bin/bash
#SBATCH -J pspe-dsweep
#SBATCH -o %x.o%A_%a
#SBATCH -e %x.e%A_%a
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --array=0-3
#
# The return/violation frontier on rdf. This is the experiment that decides
# whether "we are better" is a trade-off we control or an advantage we bought by
# accepting violations.
#
# Where it stands without this run (§3.4m, 20 seeds, rdf, delta = 0.1 only):
#
#   arm              return   violating   worst/limit   real transitions
#   Saute RL        -7.2530      0.0%        0.25x         38,400
#   CPO             -7.1984      0.0%        0.30x         38,400
#   primal-dual     -7.1993      0.0%        0.30x         38,400
#   do-nothing      -7.4001      0.0%        0.55x              0
#   PPO-Lagrangian  -6.9699      2.3%        1.22x         38,400
#   ours            -6.8995      5.0%        1.54x          8,320
#
# We beat three of the four on return, seed-paired, intervals excluding zero,
# at 4.6x fewer real transitions -- and we are the LEAST SAFE arm in the table.
# Three of them reach 0% by operating at a quarter of the limit, with mean cost
# at or below the do-nothing policy, which costs them 0.30-0.35 of return.
#
# The margin already consumes 17.9% of the budget (effective limit 2.677 against
# 3.260) and still leaves a worst excursion of 1.54x, essentially unchanged from
# the untreated arm's 1.55x: it reduces how OFTEN the limit is exceeded, not how
# badly. So "set delta lower and buy the safety back" is structurally available
# and has never been tested. delta has been 0.1 in every rdf run.
#
# PRE-REGISTERED, before the first run:
#   - realised violation should fall monotonically with delta, and at delta =
#     0.01 should approach 0%;
#   - return should fall with it, because the margin takes more of the budget;
#   - the open question is the SHAPE. If at the delta where we reach ~0% we
#     still beat CPO, primal-dual NPG and Saute on return, the claim becomes a
#     frontier that dominates their single points and is much stronger than it
#     is today. If instead the return advantage is gone by then, the honest
#     conclusion is that the advantage was bought by accepting violations, and
#     that belongs in the paper rather than in a reviewer's report.
#   - the worst-excursion ratio is the secondary outcome: if it stays near 1.5x
#     at every delta, the margin controls frequency only, and the guarantee
#     should be stated that way everywhere.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"
[ -f data/pde_testbeds/rdf_64.npz ] || { echo "missing rdf_64.npz" >&2; exit 1; }

DELTAS=(0.01 0.02 0.05 0.10)
D="${DELTAS[${SLURM_ARRAY_TASK_ID:-0}]}"
OUT="runs/delta_sweep_rdf/d${D}"
mkdir -p "$OUT"
echo "=== rdf, delta = $D, 20 seeds ===" >&2

# The published rdf configuration, with only --margin-delta varying, so the
# sweep is a sweep and not a reconfiguration.
for BATCH in "0 1 2 3 4" "5 6 7 8 9" "10 11 12 13 14" "15 16 17 18 19"; do
  for s in $BATCH; do
    python -u eval/run_constraint_fix.py --testbed rdf --seed "$s" --grid 64 \
      --epochs 20 --iterations 200 --probe-every 10 \
      --sat-coef 1.0 --adv-floor 1e-2 --dual-ema 0.7 --ki 0.5 \
      --margin-conformal --margin-delta "$D" \
      --device cuda --out "$OUT/seed_$s" > "$OUT/seed_$s.log" 2>&1 &
  done
  wait
  echo "=== delta $D batch $BATCH done ===" >&2
done
echo "=== delta $D complete: $(ls -d $OUT/seed_* 2>/dev/null | wc -l) seeds ===" >&2
