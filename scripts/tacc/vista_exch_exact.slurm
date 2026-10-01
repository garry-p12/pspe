#!/bin/bash
#SBATCH -J t21-exact
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#
# T2.1, exact version. §9.3c bounded the coverage penalty with the KS statistic
# because the earlier runs saved only summaries; KS <= d_TV, so the inequality
# ran the safe way and the conclusion held, but the reported tightness (11% of
# the drift budget) is an upper estimate rather than the real one.
#
# run_exchangeability.py now persists the ordered score sequence, which is what
# the weighted total-variation term in [Barber et al., 2023] actually needs. Both
# families, so the comparison dar-vs-rdf is like for like -- dar did not reject
# the lag-1 test and rdf did, and the bound should reflect that difference.
#
# On GPU this time: the CPU run took 4 h a seed and timed out at seed 0.
set -euo pipefail
VENV="${VENV:-$WORK/pspe-venv}"; REPO="${REPO:-$WORK/pspe}"
module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"; cd "$REPO"
# is_available() is NOT enough: on gb (Grace-Blackwell, sm_100) the device is
# visible but this venv's torch is built for gh (Grace-Hopper, sm_90), so every
# kernel launch fails with "no kernel image is available for execution on the
# device". Three jobs reported COMPLETED with exit 0 and empty output before
# this was caught. Run a real kernel, not a capability query.
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; x=torch.randn(64,64,device='cuda'); assert torch.isfinite(x@x).all(), 'CUDA kernels unusable on this node'; print('CUDA ok:', torch.cuda.get_device_name(0))"

for TB in rdf dar; do
  for SEED in 0 1 2; do
    python -u eval/run_exchangeability.py --testbed "$TB" --seeds "$SEED" \
        --grid 64 --epochs 20 --iterations 400 --probe-every 10 \
        --delta 0.1 --calib-frac 0.5 --n-perm 2000 --device cuda \
        --out "runs/exch_exact_${TB}/seed_${SEED}" \
        > "runs/exch_exact_${TB}_seed_${SEED}.log" 2>&1 &
  done
  wait
  python eval/run_exchangeability.py --aggregate --out "runs/exch_exact_${TB}" || true
  echo "=== $TB done ===" >&2
done
