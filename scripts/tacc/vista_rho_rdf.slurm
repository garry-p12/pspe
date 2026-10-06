#!/bin/bash
#SBATCH -J pspe-rho-rdf
#SBATCH -o %x.o%A_%a
#SBATCH -e %x.e%A_%a
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:40:00
#SBATCH --array=0-19
#
# Does Proposition 3's closed form predict a LEARNED planner's violation rate?
# rdf only, one seed per task.
#
# R(rho) = 1 - Phi(z_delta / rho) is validated to 0.003 in runs/margin_synthetic/,
# where sigma and epsilon are dialled by hand and nothing is learned. This run
# puts the measured rho beside the realised violation rate of the recipe the
# formula is about -- the model-error margin, which Proposition 3 excludes and
# which arms() did not contain until now.
#
# ONE SEED PER TASK rather than four batches of five. The batched version was
# estimated by the scheduler to start 16 hours out, while a 20-task array of
# single-seed 40-minute jobs backfills in minutes on a queue 238 deep -- the
# wildfire array of exactly that shape cleared within half an hour. Measured
# cost: the equivalent 4-arm rdf run did 20 seeds in 4 batches of 5 in 59:36, so
# ~15 minutes per batch of five concurrent; one seed alone with five arms should
# be well inside 40 minutes.
#
# PRE-REGISTERED, before the first run:
#   - rho > 1, because the realised-cost spread exceeds the matched-pair model
#     error wherever the dual is active;
#   - the model-error arm's realised violation sits NEAR R(rho) computed from
#     that seed's own rho, and ABOVE delta;
#   - probe+margin sits near delta and well below R(rho).
# Falsifier: the model-error arm landing far from R(rho), in which case the
# theorem holds in the Gaussian setting it was proved for and does not transfer.
# A back-of-envelope at rho = 1.40 predicts 0.180 against a previously measured
# 0.182, but that rho was chosen to produce the match, so it is a hypothesis.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"; SEED="${SLURM_ARRAY_TASK_ID:-0}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"
[ -f data/pde_testbeds/rdf_64.npz ] || { echo "missing rdf_64.npz" >&2; exit 1; }
OUT=runs/rho_test_rdf
mkdir -p "$OUT"
srun python -u eval/run_constraint_fix.py --testbed rdf --seed "$SEED" --grid 64 \
    --epochs 20 --iterations 200 --probe-every 10 \
    --sat-coef 1.0 --adv-floor 1e-2 --dual-ema 0.7 --ki 0.5 \
    --margin-conformal --margin-delta 0.1 \
    --device cuda --out "$OUT/seed_${SEED}"
