#!/bin/bash
#SBATCH -J pspe-margin
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#
# "What should a conformal margin bound?" -- the two experiments the ICML draft
# defers, packed onto one GH200. Each seed needs ~2 GB of 96 GB, so all seeds of
# an experiment run concurrently and the job costs wall-clock only.
#
#   sbatch -A ATM23014 --export=ALL,EXPERIMENT=sweep scripts/tacc/vista_margin.slurm
#   sbatch -A ATM23014 --export=ALL,EXPERIMENT=ndws  scripts/tacc/vista_margin.slurm
#   sbatch -A ATM23014 --export=ALL,EXPERIMENT=cap     scripts/tacc/vista_margin.slurm
#   sbatch -A ATM23014 --export=ALL,EXPERIMENT=calib   scripts/tacc/vista_margin.slurm
#   sbatch -A ATM23014 --export=ALL,EXPERIMENT=perturb scripts/tacc/vista_margin.slurm
#   sbatch -A ATM23014 --export=ALL,EXPERIMENT=perturbarms,SHIFTS='0.01 0.04 0.16' scripts/tacc/vista_margin.slurm
#   sbatch -A ATM23014 --export=ALL,EXPERIMENT=synth scripts/tacc/vista_margin.slurm
#
#   sweep  fidelity sweep on rdf: 5 surrogate qualities x 4 margin recipes x
#          3 seeds. The point is to straddle rho = sigma/eps = 1, so the two
#          weakest surrogates are there to reach rho < 1 where the default
#          should be correct; without them the sweep only restates the failure.
#   ndws   the margin on real fire dynamics under an expectation constraint,
#          which is the regime the matched-pair recipe cannot cover. 3 seeds.
#   synth  the controlled coverage study. Seconds on a CPU; here only so the
#          whole claim can be reproduced from one script.
set -uo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}" HF_HUB_OFFLINE=1
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
DEV="${DEVICE:-cuda}"
if [ "$DEV" = cuda ]; then
  python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
SEEDS="${SEEDS:-0 1 2}"
mkdir -p runs/cap runs/margin_choice runs/margin_choice_hires runs/ndws_margin runs/margin_synthetic runs/ndws_calib runs/ndws_perturb

case "${EXPERIMENT:?set EXPERIMENT=sweep|ndws|synth}" in
  sweep)
    # Fidelity is swept by training budget: "epochs:fraction-of-trajectories".
    # 1:0.1 gives rel L2 ~0.57 against ~0.002 at 20:1.0, which spans the regime
    # where the default should win into the one where it fails.
    #
    # One process per (seed, fidelity) rather than one per seed. Inside a
    # process the four margin arms are sequential, so packing only over seeds
    # would make wall-clock 3 x 4 = 12 arms deep and overrun the time limit;
    # this way it is 4 deep and the GPU carries 9 small jobs, each a couple of
    # GB of 96.
    for s in $SEEDS; do
      # The two weakest levels exist because of a check that nearly went
      # unmade: at 1:0.1 the measured rho = sigma/eps is already 1.35, and rho
      # only RISES as the surrogate improves. A sweep starting there would
      # visit the failure regime exclusively and could never show the default
      # working, which is the half of the claim that makes it a diagnostic
      # rather than a complaint. 1:0.02 and 1:0.05 push eps above sigma.
      # A ladder in BOTH knobs. Subsetting alone cannot span the range: rdf
      # has 96 trajectories, so every fraction below ~0.09 clamped to the old
      # floor of 8 and produced the same surrogate under a different name.
      for f in ${FIDELITIES:-1:0.08 2:0.2 5:0.4 10:0.7 20:1.0}; do
        tag="${f//[:.]/_}"
        # EVAL_EVERY 5 gives ~41 evaluations per run against 11 at the default,
        # so the violation rate can resolve a delta of 0.1 instead of landing
        # on multiples of 1/11 either side of it.
        python eval/run_margin_choice.py --testbed "${TESTBED:-rdf}" --grid 64 \
          --iterations "${ITERATIONS:-200}" --probe-every 20 --probe-episodes 16 \
          --eval-every "${EVAL_EVERY:-20}" \
          --fidelity "$f" --margin-delta 0.1 \
          --seed "$s" --device "$DEV" \
          --out "runs/margin_choice${RUNSFX:-}/${TESTBED:-rdf}_s${s}_f${tag}" \
          > "runs/margin_choice${RUNSFX:-}/${TESTBED:-rdf}_s${s}_f${tag}.log" 2>&1 &
      done
    done ;;
  ndws)
    # G is the planner's surrogate (short training); F stands in for reality
    # (longer, wider, different seed). Both are fitted to observed fire days.
    for s in $SEEDS; do
      python eval/run_ndws_margin.py --n-train 8000 --n-eval 1500 \
        --weak-epochs 4 --real-epochs 18 --iterations 60 --plan-steps 40 \
        --plan-batch 32 --eval-batch 32 --cal-fires 256 --cal-lambda 50 \
        --probe-every 5 --margin-delta 0.1 \
        --seed "$s" --device "$DEV" --out "runs/ndws_margin/seed$s" \
        > "runs/ndws_margin/seed$s.log" 2>&1 &
    done ;;
  calib)
    # Feasibility only: does a configuration leave room for a margin at all?
    # Three quantities decide it, and all three come out of calibration in a
    # couple of minutes:
    #   span      how far the planner can move the constrained quantity
    #   bias      how far reality sits from where the dual thinks it is
    #   headroom  how much of the limit a margin may consume
    # A margin must cover the bias and fit inside the headroom. At budget 3%
    # the measured bias was 1.4x the entire span, so no limit worked. More
    # crew widens the span; a better G narrows the bias; both are swept.
    for cfg in "0.03 4" "0.08 4" "0.08 10" "0.15 10"; do
      set -- $cfg; B=$1; WE=$2
      python eval/run_ndws_margin.py --calibrate-only --budget "$B" --weak-epochs "$WE" \
        --limit-frac "${LIMIT_FRAC:-0.5}" --n-train 8000 --n-eval 1500 \
        --plan-steps 40 --eval-batch 32 --cal-fires 192 --seed "${SEEDS%% *}" \
        --device "$DEV" --out "runs/ndws_calib/b${B}_e${WE}" \
        > "runs/ndws_calib/b${B}_e${WE}.log" 2>&1
      echo "--- budget $B, G epochs $WE ---"; tail -4 "runs/ndws_calib/b${B}_e${WE}.log"
    done ;;
  perturb)
    # The experiment the learned-vs-learned design could not support. Reality
    # is the planner's own model with a controlled discrepancy, so the
    # systematic bias b can be walked across the feasibility threshold b < s
    # while the per-instance error eps is held fixed. G is cached, so every
    # point after the first retrains nothing.
    #
    # Stage 1 finds where the threshold sits; stage 2 runs the margin arms on
    # both sides of it. A margin that holds delta below the threshold and
    # fails above it demonstrates the precondition directly, which is what the
    # wildfire task can support and the validation it cannot.
    for sh in ${SHIFTS:-0.01 0.02 0.04 0.08 0.16}; do
      python eval/run_ndws_margin.py --calibrate-only \
        --perturb-shift "$sh" --perturb-scatter "${SCATTER:-0.3}" \
        --budget "${BUDGET:-0.03}" --weak-epochs "${WEAK_EPOCHS:-4}" \
        --limit-frac "${LIMIT_FRAC:-0.5}" --n-train 8000 --n-eval 1500 \
        --plan-steps 40 --eval-batch 32 --cal-fires 192 --seed "${SEEDS%% *}" \
        --device "$DEV" --out "runs/ndws_perturb/calib_sh${sh}" \
        > "runs/ndws_perturb/calib_sh${sh}.log" 2>&1
      echo "--- shift $sh ---"; grep -E 'span 0|VERDICT' "runs/ndws_perturb/calib_sh${sh}.log"
    done ;;
  perturbarms)
    # Stage 2: the four margin recipes at each shift, on both sides of b = s.
    for s in $SEEDS; do
      for sh in ${SHIFTS:-0.01 0.04 0.16}; do
        python eval/run_ndws_margin.py \
          --perturb-shift "$sh" --perturb-scatter "${SCATTER:-0.3}" \
          --budget "${BUDGET:-0.03}" --weak-epochs "${WEAK_EPOCHS:-4}" \
          --limit-frac "${LIMIT_FRAC:-0.5}" --n-train 8000 --n-eval 1500 \
          --iterations 60 --plan-steps 40 --plan-batch 32 --eval-batch 32 \
          --cal-fires 192 --probe-every 5 --probe-batches 4 --margin-delta 0.1 \
          ${ARMS:+--arms $ARMS} \
          --seed "$s" --device "$DEV" --out "runs/ndws_perturb/s${s}_sh${sh}" \
          > "runs/ndws_perturb/s${s}_sh${sh}.log" 2>&1 &
      done
    done ;;
  cap)
    # The model-based safe-RL baseline. Every other safe-RL baseline in this
    # repo is model-free, which makes a sample-efficiency comparison against
    # them close to tautological for a model-based planner. CAP plans in a
    # learned model and corrects for it being wrong, which is the thing our
    # margin does, so it is the comparison that carries information.
    #
    # Same probe budget, same planner, same dual, same limit, same evaluation
    # protocol as our arms: only the correction mechanism differs.
    for s in $SEEDS; do
      python eval/run_cap_baseline.py --testbed "${TESTBED:-rdf}" --grid 64 \
        --epochs 20 --members "${MEMBERS:-5}" --iterations 200 \
        --probe-every 20 --probe-episodes 16 --eval-every 5 \
        --seed "$s" --device "$DEV" --out "runs/cap/${TESTBED:-rdf}_s${s}" \
        > "runs/cap/${TESTBED:-rdf}_s${s}.log" 2>&1 &
    done ;;
  synth)
    python eval/run_margin_synthetic.py --trials 2000 --n-cal 200 --n-test 20000 \
      --rho 0.1 0.25 0.5 0.8 1.0 1.25 2.0 4.0 10.0 \
      --out runs/margin_synthetic > runs/margin_synthetic/run.log 2>&1 &
    ;;
esac

# Collect whatever the python processes returned, so a real failure is still
# visible. Without this the summary below decides the exit status instead.
rc=0
for pid in $(jobs -p); do wait "$pid" || rc=1; done

echo "=== done: $EXPERIMENT (python rc=$rc) ==="
for f in runs/cap/*/results.md runs/margin_choice*/*/results.md \
         runs/ndws_margin/*/results.md runs/ndws_perturb/*/results.md \
         runs/margin_synthetic/results.md; do
  [ -f "$f" ] && { echo "--- $f ---"; cat "$f"; }
done

# The summary loop must not set the exit status. Its final `[ -f ]` returns 1
# whenever the last glob matches nothing, which marked five completed jobs as
# FAILED while their results sat on disk.
exit "$rc"
