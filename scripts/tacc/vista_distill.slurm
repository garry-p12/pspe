#!/bin/bash
#SBATCH -J pspe-distill
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 05:00:00
#
# The distillation test section 9.7 says we skipped.
#
# Section 9.7 concedes the amortisation claim is the weakest thing in the paper:
# a model-based planner CAN be amortised into a compact policy via behaviour
# cloning on planner-generated data [Byravan et al., 2022], and we never tried
# it -- we trained a CNN from scratch under a budget dual and reported the gap.
# So the 36.5% planner against 10.4% amortised may be a fact about our recipe.
#
# Everything but the learning signal is held fixed: the same GaussianFieldPolicy
# class the from-scratch arm uses, demonstrations collected along the planner's
# own trajectory on TRAINING patches, scored on held-out patches by the same
# score_policy every other arm in section 5.1 goes through.
#
# 3 seeds. The decisive number is how much of the planner's advantage one
# forward pass recovers; the bar to clear is the 10.4% scratch arm.
#
# Sizing: the cost is demonstration collection -- 256 fires x 50 projected
# gradient steps through the surrogate rollout, per seed. The planning arm in
# runs/ndws_plan_b* did 50 steps over 256 held-out patches inside its own run,
# so one seed is roughly that plus a cheap BC fit. 5 h for 3 seeds is the margin.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"
[ -d data/ndws ] || { echo "missing NDWS at data/ndws" >&2; exit 1; }

for SEED in 0 1 2; do
  echo "=== distill seed $SEED ===" >&2
  srun python eval/run_ndws_distill.py \
      --seed "$SEED" --device cuda \
      --grid 64 --patches 8 --horizon 5 --budget 0.03 \
      --n-train 1024 --n-eval 256 --epochs 18 \
      --plan-steps 50 --demo-fires 256 \
      --bc-epochs 30 --bc-lr 1e-3 \
      --out "runs/ndws_distill/s${SEED}"
done

python - <<'PY'
import json, glob
from pathlib import Path
rows = []
for f in sorted(glob.glob("runs/ndws_distill/s*/results.json")):
    d = json.loads(Path(f).read_text())
    rows.append(d["summary"])
Path("runs/ndws_distill/all.json").write_text(json.dumps(rows, indent=2))
if rows:
    pl = [r["planner %"] for r in rows]
    di = [r["distilled %"] for r in rows]
    print(f"{len(rows)} seeds")
    print(f"  planner   {sum(pl)/len(pl):.2f}  per-seed {[round(x,2) for x in pl]}")
    print(f"  distilled {sum(di)/len(di):.2f}  per-seed {[round(x,2) for x in di]}")
    print(f"  scratch amortised (5.1): 10.4")
PY
