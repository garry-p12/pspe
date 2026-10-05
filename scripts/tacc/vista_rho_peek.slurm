#!/bin/bash
#SBATCH -J pspe-rho-peek
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#
# ONE seed, on the development partition, to read rho before the 20-seed array
# gets a slot. gh is 238 jobs deep and the array has no estimated start; gh-dev
# is 21 deep.
#
# This is not the experiment -- it is the single number the experiment hinges on.
# R(rho) = 1 - Phi(z_delta / rho) was matched against a previously measured 0.182
# at rho = 1.40, but that rho was CHOSEN to produce the match. If the measured
# rho is near 1.4 the agreement is real and the 20-seed run will confirm it; if
# it is 3 or more then R(rho) ~ 0.33 and the back-of-envelope was a coincidence.
# Either way one seed settles which, and that changes how the result is framed
# rather than whether it is reported.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"
srun python -u eval/run_constraint_fix.py --testbed rdf --seed 0 --grid 64 \
    --epochs 20 --iterations 200 --probe-every 10 \
    --sat-coef 1.0 --adv-floor 1e-2 --dual-ema 0.7 --ki 0.5 \
    --margin-conformal --margin-delta 0.1 \
    --device cuda --out runs/rho_peek_rdf/seed_0
