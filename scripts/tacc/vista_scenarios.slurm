#!/bin/bash
#SBATCH -J pspe-scen
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#
# Rebuild the portal's scenario library with a CALIBRATION SET, not a handful
# of illustrative combinations.
#
# The portal currently shows a plus/minus band that is a normal-approximation
# prediction interval on the surrogate's residuals. The framework's actual
# contribution is a split-conformal quantile with a stated failure rate, and
# that needs n >= 1/delta - 1 calibration points: 9 for delta = 0.1, 19 for
# 0.05. The old library had six multi-measure runs, so the margin function
# correctly refused to state 90% and the portal fell back on a Gaussian
# assumption it has no right to.
#
# This enumerates 30 multi-measure combinations at varied heights -- varied
# because the calibration sample has to be exchangeable with what a planner
# actually proposes, and a planner spending a budget picks partial heights far
# more often than full ones. All scenarios share one batched solve, so 37
# options cost roughly what 13 did.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
ROOT="${ROOT:-$WORK/data/floodcastbench/extracted}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
[ -d "$ROOT" ] || { echo "missing floodcastbench at $ROOT" >&2; exit 1; }

srun python eval/build_scenario_library.py --root "$ROOT" --event australia2022 \
    --coarsen 2 --start-hours 48 --end-hours 120 \
    --n-sites 6 --max-height 3.0 --site-width-m 900 \
    --device cuda --out runs/scenarios_v2
