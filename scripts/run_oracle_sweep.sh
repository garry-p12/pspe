#!/usr/bin/env bash
# Hindsight-oracle headroom, one process per environment.
#
# The CEM budget is set per TIER from measured ms/step, not globally: 128x12
# is ten minutes on `FloodCrossing-v0` and ten hours on `FloodGreenInfra-v0`.
# A tier's budget is recorded in its own oracle.json, because an oracle run at
# 16x4 is a weaker lower bound on the ceiling than one run at 128x12 and the
# two must not be read as the same quantity.
#
#   scripts/run_oracle_sweep.sh cheap      # <=25 ms/step, 128x12
#   scripts/run_oracle_sweep.sh mid        # <=200 ms/step, 64x8
#   scripts/run_oracle_sweep.sh slow       # the rest, 24x6
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/opt/anaconda3/envs/pspe/bin/python}
TIER=${1:-cheap}
PARALLEL=${PARALLEL:-6}
mkdir -p runs/gym/oracle_logs

case "$TIER" in
  cheap) ENVS=(FloodCrossing-v0 FloodEvacuate-v0 CMDPToy-v0
               PDEControl-dar-v0 PDEControl-rdf-v0 PDEControl-swe-v0)
         CANDS=128; ITERS=12 ;;
  mid)   ENVS=(HeatRetrofit-v0 Firebreak-v0 FloodOperate-v0 FireDispatch-v0)
         CANDS=64; ITERS=8 ;;
  slow)  ENVS=(FloodBuyout-v0 FloodOperateFine-v0 FloodDefend-v0
               FloodGreenInfra-v0)
         CANDS=24; ITERS=6 ;;
  *) echo "tier must be cheap, mid or slow"; exit 2 ;;
esac

printf '%s\n' "${ENVS[@]}" | xargs -P "$PARALLEL" -I{} \
  "$PY" scripts/gym_oracle.py --env {} --cands "$CANDS" --iters "$ITERS" \
       --out "runs/gym/oracle_{}.json" \
  2>&1 | tee "runs/gym/oracle_logs/${TIER}.log"
echo "tier $TIER done at ${CANDS}x${ITERS}"
