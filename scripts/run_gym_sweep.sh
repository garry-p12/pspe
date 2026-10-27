#!/usr/bin/env bash
# Train a constrained policy on every catalog environment, three seeds each.
#
# One process per (environment, seed) pair, because none of these environments
# vectorises -- `FloodDefend-v0` wraps a LISFLOOD solver and `Firebreak-v0`
# wraps a CNN, so the only parallelism available is across runs. Each process
# pins torch to one thread (the trainer does this itself); PARALLEL stays below
# the core count so the slow solvers are not fighting each other for a core.
#
# Jobs are emitted longest-first. `FloodGreenInfra-v0` is ~0.85 s per step with
# the policy in the loop, so starting it last would leave one core running for
# an hour after everything else finished.
#
#   scripts/run_gym_sweep.sh                      # all fourteen, seeds 0 1 2
#   PARALLEL=4 SEEDS="0 1" scripts/run_gym_sweep.sh
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/opt/anaconda3/envs/pspe/bin/python}
PARALLEL=${PARALLEL:-7}
SEEDS=${SEEDS:-"0 1 2"}
ROOT=${ROOT:-runs/gym}
LOGS=${LOGS:-runs/gym/logs}
mkdir -p "$LOGS"

# Descending wall-clock, measured on this machine with the policy in the loop.
ENVS=(
  FloodGreenInfra-v0 FloodOperateFine-v0 FloodDefend-v0 FloodBuyout-v0
  FireDispatch-v0 FloodOperate-v0 Firebreak-v0 HeatRetrofit-v0
  PDEControl-swe-v0 PDEControl-rdf-v0 PDEControl-dar-v0
  CMDPToy-v0 FloodEvacuate-v0 FloodCrossing-v0
)

jobs_file=$(mktemp)
for e in "${ENVS[@]}"; do
  for s in $SEEDS; do echo "$e $s"; done
done > "$jobs_file"
echo "$(wc -l < "$jobs_file") jobs, $PARALLEL at a time, logs in $LOGS"

run_one() {
  set -- $1
  local env=$1 seed=$2
  local log="$LOGS/${env}.seed${seed}.log"
  if [ -f "$ROOT/$env/seed$seed/result.json" ]; then
    echo "skip $env seed $seed (already finished)"; return 0
  fi
  "$PY" scripts/train_gym_policy.py --env "$env" --seed "$seed" \
      --eval-every 5 --root "$ROOT" > "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then echo "done $env seed $seed -- $(tail -1 "$log")"
  else echo "FAIL $env seed $seed rc=$rc -- see $log"; fi
  return 0
}
export -f run_one
export PY ROOT LOGS

xargs -P "$PARALLEL" -I{} bash -c 'run_one "$@"' _ {} < "$jobs_file"
rm -f "$jobs_file"

echo
"$PY" scripts/train_gym_policy.py --report --root "$ROOT"
