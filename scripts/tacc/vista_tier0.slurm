#!/bin/bash
#SBATCH -J pspe-tier0
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#
# Tier 0 of the ICML revision plan (section 8.5): the two things a reviewer can
# check in five minutes and that we currently fail.
#
# (a) EXCHANGEABILITY ON rdf. The reviewer's exact words: "the empirical
#     exchangeability check is only on DAR -- not RDF, despite RDF carrying much
#     of the headline margin evidence." Confirmed: runs/exchangeability/ is dar
#     only. rdf is the family every margin number in section 3.2 comes from, so
#     the assumption is untested precisely where it is load-bearing.
#
# (b) TWENTY SEEDS for the two comparative claims our own power audit (3.2c)
#     found cannot separate from zero at five:
#       constraint_fix_sat_rdf  43.6% -> 32.7%, CI [-30.3, +12.1]   (section 3.3)
#       alpha_rule              variance rule vs fixed, CI [-12.1, +6.1] (4.2)
#     Section 3.2b showed what happens when this is done properly: an effect
#     that five seeds could not see turned out to be real and larger than
#     published. These two deserve the same test.
#
# Configurations are copied verbatim from the published runs (vista_packed.slurm
# arms `cfixconf`, `cfixsat`, `alpha`) so these extend those experiments rather
# than starting different ones.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"

# --- (a) exchangeability on rdf, 3 seeds, same sizing as the dar run ---------
echo "=== exchangeability: rdf ===" >&2
for SEED in 0 1 2; do
  srun python eval/run_exchangeability.py --testbed rdf --seeds "$SEED" \
      --grid 64 --epochs 20 --iterations 400 --probe-every 10 \
      --delta 0.1 --calib-frac 0.5 --n-perm 2000 --device cuda \
      --out "runs/exchangeability_rdf/seed_${SEED}"
done
python eval/run_exchangeability.py --aggregate --out runs/exchangeability_rdf || true

# --- (b) twenty seeds, section 3.3's planner-defect claim --------------------
echo "=== constraint_fix_sat_rdf: 20 seeds ===" >&2
for BATCH in "0 1 2 3 4" "5 6 7 8 9" "10 11 12 13 14" "15 16 17 18 19"; do
  for s in $BATCH; do
    python -u eval/run_constraint_fix.py --testbed rdf --seed "$s" --grid 64 \
      --epochs 20 --iterations 200 --probe-every 20 --margin-k 2.0 \
      --sat-coef 1.0 --adv-floor 1e-2 --dual-ema 0.7 --device cuda \
      --out "runs/constraint_fix_sat_rdf_20/seed_$s" \
      > "runs/constraint_fix_sat_rdf_20_seed_$s.log" 2>&1 &
  done
  wait
done

# --- (b) twenty seeds, section 4.2's mixing rule -----------------------------
echo "=== alpha_rule: 20 seeds ===" >&2
for BATCH in "0 1 2 3 4" "5 6 7 8 9" "10 11 12 13 14" "15 16 17 18 19"; do
  for s in $BATCH; do
    python -u eval/run_alpha_rule.py --seed "$s" --grid 64 --epochs 20 \
      --iterations 200 --device cuda --out "runs/alpha_rule_20/seed_$s" \
      > "runs/alpha_rule_20_seed_$s.log" 2>&1 &
  done
  wait
done

# --- the seed-paired bootstraps these runs exist to make possible ------------
python eval/run_bootstrap_report.py --trees "sat_rdf20=runs/constraint_fix_sat_rdf_20" \
    --out runs/bootstrap_report_sat20 || true
echo "=== tier 0 complete ===" >&2
