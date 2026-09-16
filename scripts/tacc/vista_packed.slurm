#!/bin/bash
#SBATCH -J pspe-packed
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#
# All seeds of one experiment concurrently on ONE GH200, instead of one node
# per seed. When the partition is saturated (573/576 nodes, ~480 jobs ahead),
# a single-node short job backfills into gaps that a 5-node array never fits.
# Each run needs ~2 GB of a 96 GB GPU, so packing costs nothing but wall-clock.
#
#   sbatch -A <alloc> --export=ALL,EXPERIMENT=cfix scripts/tacc/vista_packed.slurm
#   EXPERIMENT in: cfix | expl | res
set -uo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}" HF_HUB_OFFLINE=1
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

case "${EXPERIMENT:?set EXPERIMENT=cfix|expl|res}" in
  cfix)
    for s in 0 1 2 3 4; do
      python eval/run_constraint_fix.py --seed $s --grid 64 --epochs 20 --iterations 200 \
        --probe-every 20 --margin-k 2.0 --device cuda --out runs/constraint_fix/seed_$s \
        > runs/constraint_fix_seed_$s.log 2>&1 &
    done ;;
  expl)
    for s in 0 1 2; do
      ( CKPT="runs/explain_real_v2/seed_$s/policy.pt"
        python scripts/train_plan.py seed=$s device=cuda planner.iterations=200 \
          output.checkpoint="$CKPT" hydra.run.dir="outputs/expl_v2_plan_$s" &&
        python eval/run_explain_baselines.py --testbed dar --grid 64 --iterations 200 \
          --device cuda --backbone Qwen/Qwen2.5-0.5B-Instruct --policy "$CKPT" \
          --seed $s --out runs/explain_real_v2/seed_$s
      ) > runs/explain_v2_seed_$s.log 2>&1 &
    done ;;
  res)
    for s in 0 1 2 3 4; do
      python eval/run_resolution.py --testbed dar --train-grid 64 --eval-grids 64 96 128 \
        --epochs 20 --seed $s --device cuda --out runs/resolution_seeds/seed_$s \
        > runs/resolution_seed_$s.log 2>&1 &
    done ;;
esac
wait
echo "packed $EXPERIMENT done"
