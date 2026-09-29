#!/bin/bash
#SBATCH -J pspe-smbpo
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#
# The second model-based safe-RL baseline, which section 9.2 lists as unrun.
#
# CAP is already transplanted onto our planner: 11.99% violating against our
# 7.80%, z = +2.08. Section 3.2 explains that structurally -- ensemble
# disagreement is not the distribution that breaches the limit -- and a
# structural claim is only worth the name if it predicts the NEXT method too.
#
# SMBPO corrects by pessimism (max over the ensemble, truncated horizon) rather
# than CAP's adaptive mean + k*std. Both draw their correction from model
# disagreement, so section 3.2 predicts SMBPO also loses to the conformal
# margin, and by MORE than CAP: max is a cruder statistic and there is no
# adaptation to walk it back when disagreement is uninformative. That
# prediction is recorded in eval/run_smbpo_baseline.py before the run.
#
# 5 seeds on rdf, the family where the constraint actually binds (worst cost
# 4.515 against a 3.26 limit). dar is carried as the regression check that a
# method does not break the family where nobody violates.
#
# Sizing: CAP's run is 5 surrogate members plus 200 planner iterations. SMBPO
# adds a truncated-horizon ensemble pass per dual step, which is cheaper than
# CAP's full-horizon one, so per seed it should come in under CAP. 4 h for 10
# runs is the margin.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"

for SEED in 0 1 2 3 4; do
  for TB in rdf dar; do
    echo "=== SMBPO $TB seed $SEED ===" >&2
    srun python eval/run_smbpo_baseline.py \
        --testbed "$TB" --seed "$SEED" --device cuda \
        --members 5 --epochs 20 --iterations 200 --horizon 12 \
        --imagine-frac 0.5 \
        --probe-every 20 --probe-episodes 16 --eval-every 5 \
        --out "runs/smbpo/${TB}_s${SEED}"
  done
done

# One aggregate so the comparison against CAP and our margin is a single table.
python - <<'PY'
import json, glob
from pathlib import Path
rows = []
for f in sorted(glob.glob("runs/smbpo/*/results.json")):
    rows.extend(json.loads(Path(f).read_text()))
Path("runs/smbpo/all.json").write_text(json.dumps(rows, indent=2))
for tb in ("rdf", "dar"):
    sel = [r for r in rows if r["testbed"] == tb]
    if not sel:
        continue
    v = [r["violating"] for r in sel]
    print(f"{tb}: {len(sel)} seeds, violating mean {sum(v)/len(v):.4f}, "
          f"per-seed {[round(x,4) for x in v]}")
PY
