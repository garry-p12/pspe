#!/bin/bash
#SBATCH -J pspe-wsts
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#
# The twin loop on WildfireSpreadTS: 607 fires where section 5.2 used 8.
#
# Held out by YEAR, not by fire. Leave-one-fire-out trains on other fires from
# the same season and weather regime, so it leaks; the dataset's own
# documentation recommends cross-validation across years because the
# distribution shifts between them. It is the stricter split and the one
# reviewers will expect, so the headline may well come DOWN -- that is the
# point of running it.
#
# Sizing, measured rather than guessed (four earlier timeouts were guesses):
#   load        6.5 min for 13,607 GeoTIFFs at 34.7 files/s, once
#   fits        4 year folds x 3 seeds = 12, each on ~450 fires
#   scoring     607 fires x 3 modes x 3 seeds
# Estimated ~100 min total; 4 h is a 2.4x margin, all three seeds in one job so
# the archive is read once.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
WSTS="${WSTS:-$WORK/data/wsts/WildfireSpreadTS}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"
[ -d "$WSTS/2018" ] || { echo "missing WildfireSpreadTS at $WSTS" >&2; exit 1; }

srun python eval/run_firms_twin.py --source wsts --wsts-root "$WSTS" \
    --cv year --grid 64 --epochs 60 --width 24 --horizon 3 \
    --adapt-lr 0.02 --seeds 0 1 2 --device cuda \
    --out runs/wsts_twin
