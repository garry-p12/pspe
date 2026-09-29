#!/bin/bash
#SBATCH -J pspe-conf20
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#
# Twenty seeds for the conformal margin on rdf, because five could not settle it.
#
# Section 3.2a: the headline safety claim (14.5% -> 7.3% violating) is, when
# differenced within seed and bootstrapped over seeds, -6.1 points with a 95% CI
# of [-21.2, +6.1]. Three of five seeds improve, one gets worse. The effect and
# its uncertainty are the same size, so the comparative claim -- the paper's
# central safety contribution -- is currently unsupported.
#
# Five seeds cannot fix that by being re-analysed; the interval is wide because
# n is 5. Twenty seeds narrows it by about half if the effect is real, and kills
# it honestly if it is not. Either outcome is publishable; the present one is not.
#
# Same configuration as the published `cfixconf` arm in vista_packed.slurm, so
# this extends that experiment rather than starting a different one:
#   --probe-every 10 --sat-coef 1.0 --adv-floor 1e-2 --dual-ema 0.7 --ki 0.5
#   --margin-conformal --margin-delta 0.1
# Seeds 0-19 are ALL re-run into a fresh tree rather than topping up the existing
# five, so the 20 are one self-consistent sample with no provenance to track.
#
# Each seed trains four arms (baseline, probe, margin, probe+margin) for 200
# iterations. Five run concurrently per batch, as the packed script does.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"

OUT=runs/constraint_fix_conf_rdf_20
mkdir -p "$OUT"
for BATCH in "0 1 2 3 4" "5 6 7 8 9" "10 11 12 13 14" "15 16 17 18 19"; do
  echo "=== batch $BATCH ===" >&2
  for s in $BATCH; do
    python -u eval/run_constraint_fix.py --testbed rdf --seed "$s" --grid 64 \
      --epochs 20 --iterations 200 --probe-every 10 \
      --sat-coef 1.0 --adv-floor 1e-2 --dual-ema 0.7 --ki 0.5 \
      --margin-conformal --margin-delta 0.1 --device cuda \
      --out "$OUT/seed_$s" > "$OUT/seed_$s.log" 2>&1 &
  done
  wait
  echo "=== batch $BATCH done ===" >&2
done

# The seed-paired bootstrap this run exists to make possible.
python eval/run_bootstrap_report.py --trees "rdf20=$OUT" --out runs/bootstrap_report_20 || true
