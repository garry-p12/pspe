#!/bin/bash
# Laptop-side: wait for the queued Vista jobs, then pull results and aggregate
# locally. Detached with nohup so it survives the session that launched it.
#
#   nohup bash scripts/tacc/fetch_when_done.sh > runs/fetch.log 2>&1 &
#
# Complements the cluster-side watcher (tmux `pspewatch`), which aggregates on
# Vista regardless of whether this laptop is connected. This script exists so
# the results also land here without anyone remembering to fetch them; if the
# socket dies mid-wait it exits cleanly and the cluster-side copy is still there.
set -uo pipefail
cd "$(dirname "$0")/../.."
HOST="${HOST:-vista2}"
PY="${PY:-/opt/anaconda3/envs/pspe/bin/python}"
JOBS="${JOBS:-1013191,1013192,1013193,1013194}"

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%SZ')] $*"; }

log "waiting on jobs $JOBS via $HOST"
while true; do
    n=""
    for attempt in 1 2 3 4 5 6; do   # a single failed poll is usually a login-node hiccup, not a dead socket
        n=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" "squeue -j $JOBS -h 2>/dev/null | wc -l" 2>/dev/null | tr -d ' ')
        [ -n "$n" ] && break
        log "poll failed (attempt $attempt/6), retrying in 60s"
        sleep 60
    done
    if [ -z "$n" ]; then
        log "ssh failed 6 times — socket gone. Cluster-side watcher still aggregates; rerun this after re-auth."
        exit 1
    fi
    [ "$n" = "0" ] && break
    log "$n task(s) still queued/running"
    sleep 300
done
log "queue empty — fetching"

for d in constraint_fix_sat constraint_fix_sat_rdf; do
    rsync -az "$HOST":"\$WORK/pspe/runs/$d/" "runs/$d/" && log "fetched $d"
done

# Aggregate locally with the key each runner uses.
"$PY" eval/run_seeds.py --aggregate-only --seeds 0 1 2 --out runs/conformal_real --key delta >/dev/null 2>&1 && log "aggregated conformal_real"
"$PY" eval/run_seeds.py --aggregate-only --seeds 0 1 2 --out runs/faith_weights --key arm >/dev/null 2>&1 && log "aggregated faith_weights"
"$PY" eval/run_seeds.py --aggregate-only --seeds 0 1 2 3 4 --out runs/lipschitz --key arm >/dev/null 2>&1 && log "aggregated lipschitz"
"$PY" eval/run_seeds.py --aggregate-only --seeds 0 1 2 3 4 --out runs/alpha_rule --key arm >/dev/null 2>&1 && log "aggregated alpha_rule"
"$PY" eval/run_seeds.py --aggregate-only --seeds 0 1 2 3 4 --out runs/joint --key arm >/dev/null 2>&1 && log "aggregated joint"

{
    echo "# Vista batch (Phase 1 + 3 testbed sweeps), fetched $(date -u '+%Y-%m-%d %H:%M:%SZ')"
    for d in constraint_fix_sat constraint_fix_sat_rdf; do
        echo; echo "## $d"; cat "runs/$d/results_seeds.md" 2>/dev/null || echo "(missing)"
    done
} > runs/VISTA_SATFIX_RESULTS.md
log "summary written to runs/VISTA_SATFIX_RESULTS.md"
