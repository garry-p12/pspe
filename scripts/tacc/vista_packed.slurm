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
#   TESTBED (cfix, joint only): dar (default) | swe | rdf
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
case "${EXPERIMENT:?set EXPERIMENT=cfix|expl|res}" in
  cfix)    # TESTBED=dar (default) writes runs/constraint_fix/; others get a suffix
    for s in 0 1 2 3 4; do
      python eval/run_constraint_fix.py --testbed "$TB" --seed $s --grid 64 --epochs 20 --iterations 200 \
        --probe-every 20 --margin-k 2.0 --device "$DEV" --out "runs/constraint_fix$SFX/seed_$s" \
        > "runs/constraint_fix${SFX}_seed_$s.log" 2>&1 &
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
