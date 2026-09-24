#!/bin/bash
#SBATCH -J pspe-packed
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#
# All seeds of one experiment concurrently on ONE GH200, instead of one node
# per seed. When the partition is saturated (573/576 nodes, ~480 jobs ahead),
# a single-node short job backfills into gaps that a 5-node array never fits.
# Each run needs ~2 GB of a 96 GB GPU, so packing costs nothing but wall-clock.
#
#   sbatch -A <alloc> --export=ALL,EXPERIMENT=cfix scripts/tacc/vista_packed.slurm
#   sbatch -A <alloc> --export=ALL,EXPERIMENT=cfix,TESTBED=swe scripts/tacc/vista_packed.slurm
#   EXPERIMENT in: cfix | expl | expl2 | lip | alpha | joint | res
#   TESTBED (cfix, cfixsat, joint): dar (default) | swe | rdf
set -uo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}" HF_HUB_OFFLINE=1
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
DEV="${DEVICE:-cuda}"   # DEVICE=cpu for the gg (Grace, no GPU) partition
if [ "$DEV" = cuda ]; then
  python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
  # nproc honours the task cpuset (1 with -n 1); SLURM_CPUS_ON_NODE is the node.
  NT=$(( ${SLURM_CPUS_ON_NODE:-72} / 5 )); [ "$NT" -lt 1 ] && NT=1
  export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT
  echo "cpu run: ${SLURM_CPUS_ON_NODE:-?} cores, $NT threads per seed"
fi

TB="${TESTBED:-dar}"; SFX=""; [ "$TB" = dar ] || SFX="_$TB"
SFX="$SFX${RUNSFX:-}"   # RUNSFX=_gpu etc. keeps racing copies in separate directories
case "${EXPERIMENT:?set EXPERIMENT=cfix|expl|res}" in
  cfix)    # TESTBED=dar (default) writes runs/constraint_fix/; others get a suffix
    for s in 0 1 2 3 4; do
      python eval/run_constraint_fix.py --testbed "$TB" --seed $s --grid 64 --epochs 20 --iterations 200 \
        --probe-every 20 --margin-k 2.0 --device "$DEV" --out "runs/constraint_fix$SFX/seed_$s" \
        > "runs/constraint_fix${SFX}_seed_$s.log" 2>&1 &
    done ;;
  cfixsat) # rdf planner fix: saturation wall + advantage floor + faster dual; TESTBED=rdf|dar
    for s in 0 1 2 3 4; do
      python eval/run_constraint_fix.py --testbed "$TB" --seed $s --grid 64 --epochs 20 --iterations 200 \
        --probe-every 20 --margin-k 2.0 --sat-coef 1.0 --adv-floor 1e-2 --dual-ema 0.7 \
        --device "$DEV" --out "runs/constraint_fix_sat$SFX/seed_$s" \
        > "runs/constraint_fix_sat${SFX}_seed_$s.log" 2>&1 &
    done ;;
  cfixsat2) # dual-tuning variants on top of cfixsat; VARIANT selects extra flags, OUT dir suffix
    case "${VARIANT:?norm|norm_warm|norm_lowlr}" in
      norm)       EXTRA="--dual-normalize" ;;
      norm_warm)  EXTRA="--dual-normalize --lambda-init 0.3" ;;
      norm_lowlr) EXTRA="--dual-normalize --lr-policy 1e-4" ;;
      ki10)       EXTRA="--ki 0.5" ;;                       # integral holds lambda at equilibrium
      ki10_lowlr) EXTRA="--ki 0.5 --lr-policy 1e-4" ;;
      ki10_epstd) EXTRA="--ki 0.5 --margin-episode-std" ;;  # margin covers episode spread too
    esac
    for s in 0 1 2 3 4; do
      python eval/run_constraint_fix.py --testbed "$TB" --seed $s --grid 64 --epochs 20 --iterations 200 \
        --probe-every 20 --margin-k 2.0 --sat-coef 1.0 --adv-floor 1e-2 --dual-ema 0.7 $EXTRA \
        --device "$DEV" --out "runs/constraint_fix_${VARIANT}$SFX/seed_$s" \
        > "runs/constraint_fix_${VARIANT}${SFX}_seed_$s.log" 2>&1 &
    done ;;
  base)    # four safe-RL baselines, 5 seeds; TESTBED, KI (0.05 stock | 0.5 the rdf dual fix)
    KIV="${KI:-0.05}"; KSFX=""; [ "$KIV" = 0.05 ] || KSFX="_ki$KIV"
    for s in 0 1 2 3 4; do
      python eval/run_ablations.py --testbed "$TB" --grid 64 --full --only baselines \
        --baseline-ki "$KIV" --seed $s --device "$DEV" --out "runs/baselines$SFX$KSFX/seed_$s" \
        > "runs/baselines${SFX}${KSFX}_seed_$s.log" 2>&1 &
    done ;;
  ndwsplan) # constrained firebreak planning on real NDWS data, 3 seeds packed; HORIZON (default 3)
    for s in 0 1 2; do
      python -u eval/run_ndws_planning.py --seed $s --horizon "${HORIZON:-3}" --device "$DEV" \
        --out "runs/ndws_planning/seed_$s" > "runs/ndws_planning_seed_$s.log" 2>&1 &
    done ;;
  sweauth)  # swe actuation-authority sweep: at what u_max does the task become a
            # control problem at all? Reports return separation, not just cost.
    for u in 0.04 0.10 0.20 0.35 0.50; do
      python -u scripts/calibrate_constraints.py --testbeds swe --grid 64 --horizon 12 \
        --iterations 120 --u-max $u --budget "${BUDGET:-0.10}" --device "$DEV" \
        --out "runs/calibration/swe_u$u.json" > "runs/swe_auth_u$u.log" 2>&1 &
    done ;;
  ndwstest) # NDWS on the TEST split with focal loss, the split the published number uses
    for s in 0 1 2; do
      python -u eval/run_ndws.py --epochs 18 --n-train 8000 --n-eval 1500 --seed $s \
        --models unet hybrid --published-aucpr 0.284 --test --augment \
        --pos-weight "${POSW:-9.7}" --device "$DEV" \
        --out "runs/ndws_test/seed_$s" > "runs/ndws_test_seed_$s.log" 2>&1 &
    done ;;
  ndwspart) # planning under partial observation: the twin's sync step on real data
    for s in 0 1 2; do
      python -u eval/run_ndws_partial.py --seed $s --horizon 3 --device "$DEV" \
        --unet-ckpt "runs/ndws_planning/seed_$s/unet.pt" \
        --rates ${RATES:-0.0 0.15 0.35 0.6} --n-eval "${NEVAL:-800}" \
        --out "runs/ndws_partial${OUTSFX:-}/seed_$s" > "runs/ndws_partial${OUTSFX:-}_seed_$s.log" 2>&1 &
    done ;;
  ndwsbase) # constrained-RL baselines on the wildfire task. FAIR=1 starts them at
            # an action that meets the budget, which is where PSPE's optimiser starts.
    for s in ${SEEDS:-0 1 2}; do
      python -u eval/run_ndws_baselines.py --seed $s --horizon "${HORIZON:-3}" --device "$DEV" \
        --iterations "${ITERS:-200}" ${FAIR:+--fair-init} \
        --unet-ckpt "runs/ndws_planning/seed_$s/unet.pt" \
        --out "runs/ndws_baselines${OUTSFX:-}/seed_$s" > "runs/ndws_baselines${OUTSFX:-}_seed_$s.log" 2>&1 &
    done ;;
  ndwshoriz) # horizon ablation: does planning only pay when the decision is sequential?
    for s in ${SEEDS:-0 1 2}; do
      for h in 1 2 3 5; do
        python -u eval/run_ndws_planning.py --seed $s --horizon $h --device "$DEV" \
          --unet-ckpt "runs/ndws_planning/seed_$s/unet.pt" \
          --out "runs/ndws_horiz_h$h/seed_$s" > "runs/ndws_horiz_h${h}_seed_$s.log" 2>&1
      done &
    done ;;
  ndwsplan2) # NDWS planning, parameterised: SEEDS, BUDGET, BLOCK. One job = all seeds packed.
    for s in ${SEEDS:-0 1 2}; do
      CK=""; [ -f "runs/ndws_planning/seed_$s/unet.pt" ] && CK="--unet-ckpt runs/ndws_planning/seed_$s/unet.pt"
      python -u eval/run_ndws_planning.py --seed $s --horizon 3 --device "$DEV" \
        --budget "${BUDGET:-0.03}" --block "${BLOCK:-0.9}" $CK \
        --out "runs/ndws_plan${OUTSFX:-}/seed_$s" > "runs/ndws_plan${OUTSFX:-}_seed_$s.log" 2>&1 &
    done ;;
  cfixconf) # conformal margin instead of k-sigma, on TESTBED, 5 seeds
    for s in 0 1 2 3 4; do
      python -u eval/run_constraint_fix.py --testbed "$TB" --seed $s --grid 64 --epochs 20 \
        --iterations 200 --probe-every 10 --sat-coef 1.0 --adv-floor 1e-2 --dual-ema 0.7 --ki 0.5 \
        --margin-conformal --margin-delta "${DELTA:-0.1}" --device "$DEV" \
        --out "runs/constraint_fix_conf$SFX/seed_$s" > "runs/constraint_fix_conf${SFX}_seed_$s.log" 2>&1 &
    done ;;
  ndwslong) # does the permutation gap open with a real training budget? one seed,
            # ITERS iterations, gap logged every 100 so the slope is visible.
    python -u eval/run_ndws_explain.py --seed 0 --horizon 3 --device "$DEV" \
      --arms trained-in --iterations "${ITERS:-500}" --gap-every 100 \
      --contrastive "${CW:-0}" --explain-batch 8 \
      --unet-ckpt runs/ndws_planning/seed_0/unet.pt --backbone Qwen/Qwen2.5-0.5B-Instruct \
      --out "runs/ndws_explain_long${OUTSFX:-}/seed_0" > "runs/ndws_explain_long${OUTSFX:-}.log" 2>&1 &
    ;;
  ndwsexpl) # certified briefs for the NDWS firebreak plans, 3 seeds packed (Qwen).
            # ARMS selects which arms run, so a job fits a short backfill slot:
            # "trained-in" (with the certificate) is ~21 min, the others ~11.
    for s in 0 1 2; do
      python -u eval/run_ndws_explain.py --seed $s --horizon 3 --device "$DEV" \
        --arms ${ARMS:-trained-in no-faithful post-hoc} --contrastive "${CW:-0}" \
        --unet-ckpt "runs/ndws_planning/seed_$s/unet.pt" --backbone Qwen/Qwen2.5-0.5B-Instruct \
        --out "runs/ndws_explain${OUTSFX:-}/seed_$s" > "runs/ndws_explain${OUTSFX:-}_seed_$s.log" 2>&1 &
    done ;;
  expl)
    for s in 0 1 2; do
      ( CKPT="runs/explain_real_v2/seed_$s/policy.pt"
        python scripts/train_plan.py seed=$s device="$DEV" planner.iterations=200 \
          output.checkpoint="$CKPT" hydra.run.dir="outputs/expl_v2_plan_$s" &&
        python eval/run_explain_baselines.py --testbed dar --grid 64 --iterations 200 \
          --device "$DEV" --backbone Qwen/Qwen2.5-0.5B-Instruct --policy "$CKPT" \
          --seed $s --out runs/explain_real_v2/seed_$s
      ) > runs/explain_v2_seed_$s.log 2>&1 &
    done ;;
  contrast) # does the contrastive term make briefs state-specific? CW sets its weight
    for s in 0 1 2; do
      ( CKPT="runs/explain_real_v2/seed_$s/policy.pt"
        python -u eval/run_explain_baselines.py --testbed dar --grid 64 --iterations 200 \
          --device "$DEV" --backbone Qwen/Qwen2.5-0.5B-Instruct --policy "$CKPT" \
          --arms trained-in --contrastive "${CW:-1.0}" \
          --seed $s --out "runs/explain_contrast${OUTSFX:-}/seed_$s"
      ) > "runs/explain_contrast${OUTSFX:-}_seed_$s.log" 2>&1 &
    done ;;
  perm)    # permutation control on the dar Explain arms: is F state-specific?
    for s in 0 1 2; do
      ( CKPT="runs/explain_real_v2/seed_$s/policy.pt"
        python -u eval/run_explain_baselines.py --testbed dar --grid 64 --iterations 200 \
          --device "$DEV" --backbone Qwen/Qwen2.5-0.5B-Instruct --policy "$CKPT" \
          --arms ${ARMS:-trained-in no-faithful post-hoc} \
          --seed $s --out "runs/explain_perm${OUTSFX:-}/seed_$s"
      ) > "runs/explain_perm${OUTSFX:-}_seed_$s.log" 2>&1 &
    done ;;
  expl2)   # Phase 1b + 1d on Qwen: conformal certificate + faithfulness weight sweep
    for s in 0 1 2; do
      ( CKPT="runs/explain_real_v2/seed_$s/policy.pt"
        [ -f "$CKPT" ] || python scripts/train_plan.py seed=$s device="$DEV" planner.iterations=200 \
          output.checkpoint="$CKPT" hydra.run.dir="outputs/expl2_plan_$s"
        python eval/run_conformal.py --testbed dar --grid 64 --iterations 200 --device "$DEV" \
          --backbone Qwen/Qwen2.5-0.5B-Instruct --policy "$CKPT" --n-cal 200 --n-test 400 \
          --seed $s --out runs/conformal_real/seed_$s &&
        python eval/run_explain_baselines.py --testbed dar --grid 64 --iterations 200 \
          --device "$DEV" --backbone Qwen/Qwen2.5-0.5B-Instruct --policy "$CKPT" \
          --faith-weights 10 100 1000 --seed $s --out runs/faith_weights/seed_$s
      ) > runs/expl2_seed_$s.log 2>&1 &
    done ;;
  lip)     # Phase 1a: Assumption 1 / Prop. 1 measured, 5 seeds
    for s in 0 1 2 3 4; do
      python eval/run_lipschitz.py --seed $s --grid 64 --epochs 20 --device "$DEV" \
        --out runs/lipschitz/seed_$s > runs/lipschitz_seed_$s.log 2>&1 &
    done ;;
  alpha)   # Phase 1c: fixed / variance / Eq. 8 mixing rule, 5 seeds
    for s in 0 1 2 3 4; do
      python eval/run_alpha_rule.py --seed $s --grid 64 --epochs 20 --iterations 200 \
        --device "$DEV" --out runs/alpha_rule/seed_$s > runs/alpha_seed_$s.log 2>&1 &
    done ;;
  joint)   # Phase 2a: disaggregated vs joint vs joint-unanchored, 5 seeds
    for s in 0 1 2 3 4; do
      python eval/run_joint.py --testbed "$TB" --seed $s --grid 64 --epochs 20 --iterations 200 \
        --beta 0.1 --device "$DEV" --out "runs/joint$SFX/seed_$s" > "runs/joint${SFX}_seed_$s.log" 2>&1 &
    done ;;
  res)
    for s in 0 1 2 3 4; do
      python eval/run_resolution.py --testbed dar --train-grid 64 --eval-grids 64 96 128 \
        --epochs 20 --seed $s --device "$DEV" --out runs/resolution_seeds/seed_$s \
        > runs/resolution_seed_$s.log 2>&1 &
    done ;;
esac
wait
echo "packed $EXPERIMENT done"
