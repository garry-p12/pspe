#!/bin/bash
#SBATCH -J pspe-enkf
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#
# A real data-assimilation baseline beside the gain-1 nudge.
#
# Section 5.2a can only call state sync a control, because replacing the
# belief with the observation IS assimilation with the gain set to one, and
# section 9.8 concedes as much. This runs an ensemble analysis with the error
# model restored -- forecast spread from the ensemble, observation error
# stated -- so the comparison is the same method with and without its gain
# estimated, not two different families.
#
# sigma_o is SWEPT, not fitted: VIIRS ships no per-cell error variance, and
# choosing one that makes the result come out is the failure this project
# catalogues. Gain-1 nudging is the sigma_o -> 0 corner, which doubles as the
# correctness check -- at 0.05 the arm must converge on state sync, and on the
# eight-fire smoke it did (t = -0.96, not significant).
#
# Sizing: the deterministic three-mode run over 607 fires took 61 min. The
# ensemble step is batched, so on GPU 32 members cost about what one does;
# six modes at roughly 2x gives ~2 h, and 6 h is the margin.
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
    --n-ens 32 --obs-sd 0.05 0.15 0.30 \
    --out runs/wsts_twin_enkf
