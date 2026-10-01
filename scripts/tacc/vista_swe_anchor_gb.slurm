#!/bin/bash
#SBATCH -J a1-anchor
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:30:00
#
# Audit A1, stage 1: the trained reward-greedy cost anchor for the revived swe
# spec (§3.4a: horizon 96, target 0.12, u_max 0.14, budget 0.60).
#
# SIZING, from measured runtimes rather than guessed: 20 seeds at horizon 12
# take 51 min (pspe-conf20, t0-sat20, t2-adaptive20 all 51:17-51:33), i.e. about
# 13 min per batch of 5. Horizon 96 is 8x the rollout, so a batch of 5 is ~104
# min. This stage is one batch of 3. The original combined job asked 4 h for
# stage 1 PLUS a 20-seed stage 2 that alone needs ~7 h -- it would have timed out
# exactly as t0-exch did.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"

for s in 0 1 2; do
  python -u eval/run_constraint_fix.py --testbed swe --grid 64 --epochs 20 \
    --horizon 96 --task-target 0.12 --task-umax 0.14 --task-budget 0.60 \
    --sat-coef 1.0 --adv-floor 1e-2 --dual-ema 0.7 --ki 0.5 --device cuda \
    --seed $s --iterations 200 --probe-every 10 --task-cost-limit 1e6 \
    --out "runs/swe_anchor/seed_$s" > "runs/swe_anchor_$s.log" 2>&1 &
done
wait
python - <<'PY'
import json, glob
import numpy as np
from pathlib import Path
c = [json.load(open(f))["episode_cost"]
     for f in glob.glob("runs/swe_anchor/seed_*/baseline/summary.json")]
greedy = float(np.mean(c))
dn = 0.0017                       # do-nothing cost at this spec (§3.4a)
lim = dn + 0.35 * (greedy - dn)   # the 35% convention every TASK_SPECS entry uses
Path("runs/swe_anchor/limit.txt").write_text(f"{lim:.6f}\n")
print(f"anchor: greedy {greedy:.5f}, do-nothing {dn}, limit {lim:.6f}")
PY
