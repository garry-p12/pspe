#!/bin/bash
#SBATCH -J t41-safedreamer
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:15:00
#
# T4.1 (§8.5): SafeDreamer, the third model-based safe-RL baseline, run as a
# PRE-REGISTERED TEST rather than a baseline addition.
#
# PREDICTION, recorded before the run. Proposition 3 (§3.6) says any margin
# measurable from the model-error distribution alone cannot hold delta once
# rho > 1, and Corollary 3.3 says the theorem does not RANK such methods -- so
# they should all land in the same place. Measured so far: CAP 11.99%, SMBPO
# 12.20%, indistinguishable at p = 0.83, neither separating from no margin
# (14.09%). SafeDreamer's online safety correction is also a functional of model
# uncertainty, so:
#
#     SafeDreamer should land near 12%, indistinguishable from CAP and SMBPO,
#     and roughly 7 points worse than ours at 5.00%.
#
# §9.2a's earlier prediction failed because it was written after the mechanism
# rather than derived from it; this one is derived from Proposition 3.
#
# SafeDreamer's Lagrangian + online-planning safety correction is transplanted
# the way CAP and SMBPO were: same planner, dual, testbed, limit and probe
# budget, only the correction differs. Not a reimplementation of the paper.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"

for BATCH in "0 1 2 3 4"; do
  for s in $BATCH; do
    python -u eval/run_safedreamer_baseline.py --testbed rdf --seed "$s" \
      --device cuda --members 5 --epochs 20 --iterations 200 --horizon 12 \
      --probe-every 20 --probe-episodes 16 --eval-every 5 \
      --out "runs/safedreamer/rdf_s${s}" > "runs/safedreamer_rdf_$s.log" 2>&1 &
  done
  wait
done
python - <<'PY'
import json, glob
import numpy as np
v = []
for f in sorted(glob.glob("runs/safedreamer/rdf_s*/results.json")):
    v.extend(r["violating"] for r in json.loads(open(f).read()))
if v:
    print(f"SafeDreamer rdf: {len(v)} seeds, violating mean {np.mean(v):.4f}")
    print(f"  prediction was ~0.12 (CAP 0.1199, SMBPO 0.1220, ours 0.0500)")
PY
