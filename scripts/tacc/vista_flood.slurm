#!/bin/bash
#SBATCH -J pspe-flood
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#
# Flood: the conformal margin with an accepted solver as reality (§5.6).
#
#   sbatch -A ATM23014 --export=ALL,EXPERIMENT=precond scripts/tacc/vista_flood.slurm
#   sbatch -A ATM23014 --export=ALL,EXPERIMENT=rho     scripts/tacc/vista_flood.slurm
#   sbatch -A ATM23014 --export=ALL,EXPERIMENT=arms    scripts/tacc/vista_flood.slurm
#
#   precond  the cheap gate: b + z_delta*sigma_agg < f*s, per hydrograph spread.
#            Run this first. The wildfire task failed it after a day of work.
#   rho      the diagnostic sweep. rho = sigma/epsilon must STRADDLE 1, or the
#            run only restates the failure and cannot show the default working.
#            sigma comes from --q-log-sigma, epsilon from the training budget,
#            so the ladder moves BOTH knobs -- the lesson from the rdf sweep,
#            where subsetting alone clamped several levels onto one surrogate.
#   arms     the four margin recipes at a fixed operating point, 3 seeds.
#
# Time limit is deliberately short: it is the lever that gets a job backfilled.
set -uo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
DEV="${DEVICE:-cuda}"
if [ "$DEV" = cuda ]; then
  python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
SEEDS="${SEEDS:-0 1 2}"
mkdir -p runs/flood_precond runs/flood_rho runs/flood_arms
# Collect PIDs as jobs are launched. Do NOT `wait` inside the case: that reaps
# the jobs, leaving `jobs -p` empty, so the exit-code loop below would find
# nothing and report rc=0 however many runs had failed.
PIDS=()

case "${EXPERIMENT:?set EXPERIMENT=precond|rho|arms}" in
  precond)
    # Cheap and decisive. Several hydrograph spreads, because the precondition
    # can hold at one sigma and fail at another, and the paper's claim is about
    # the regime not a single point.
    for s in ${SIGMAS:-0.10 0.20 0.30 0.45}; do
      python eval/run_flood_margin.py --calibrate-only --device "$DEV" \
        --q-log-sigma "$s" --n-train "${NTRAIN:-600}" --n-cal "${NCAL:-192}" \
        --epochs "${EPOCHS:-300}" --batch "${BATCH:-96}" \
        --out "runs/flood_precond/sigma_${s}" &
      PIDS+=($!)
    done
    ;;
  rho)
    # Both knobs. Weak surrogates (few samples, few epochs) push epsilon ABOVE
    # sigma so rho < 1; strong ones pull it below so rho > 1.
    for s in $SEEDS; do
      # The two weakest levels exist for a measured reason. At 240 samples /
      # 200 epochs the flood surrogate reaches eps ~ 0.043 against sigma ~ 0.060,
      # so rho ~ 1.4 already; rho only RISES as the surrogate improves. A ladder
      # starting at 60:40 might never reach rho < 1, and a sweep that visits only
      # the failure regime cannot show the default working -- the half that makes
      # this a diagnostic rather than a complaint. 30:15 and 45:25 push eps above
      # sigma. Note eps and sigma are NOT independent (rule 19): widening the
      # hydrograph spread moves both, so surrogate budget is the only real knob.
      for lvl in ${LEVELS:-30:15 45:25 60:40 150:100 300:200 600:300}; do
        n="${lvl%%:*}"; e="${lvl##*:}"
        python eval/run_flood_margin.py --device "$DEV" --seed "$s" \
          --n-train "$n" --epochs "$e" --n-cal "${NCAL:-192}" \
          --n-eval "${NEVAL:-192}" --batch "${BATCH:-96}" \
          --q-log-sigma "${SIGMA:-0.25}" --delta "${DELTA:-0.1}" \
          --arms residual none model_error episode \
          --out "runs/flood_rho/s${s}_n${n}_e${e}" &
        PIDS+=($!)
      done
    done
    ;;
  arms)
    for s in $SEEDS; do
      python eval/run_flood_margin.py --device "$DEV" --seed "$s" \
        --n-train "${NTRAIN:-600}" --epochs "${EPOCHS:-300}" \
        --n-cal "${NCAL:-192}" --n-eval "${NEVAL:-192}" --batch "${BATCH:-96}" \
        --q-log-sigma "${SIGMA:-0.25}" --delta "${DELTA:-0.1}" \
        --arms residual none model_error episode \
        --out "runs/flood_arms/seed${s}" &
      PIDS+=($!)
    done
    ;;
  *) echo "unknown EXPERIMENT" >&2; exit 2 ;;
esac

# Collect exit codes from the recorded PIDs. Two bugs are being avoided here:
# taking rc from the last command of a reporting loop (which once marked five
# COMPLETED jobs FAILED), and waiting before collecting (which reaps the jobs
# and silently reports success).
rc=0
for p in "${PIDS[@]}"; do wait "$p" || rc=$?; done
echo "flood ${EXPERIMENT} done rc=$rc"
exit "$rc"
