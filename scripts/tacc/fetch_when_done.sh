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
JOBS="${JOBS:-998919,998920,998921}"

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

for d in constraint_fix explain_real_v2 resolution_seeds; do
    rsync -az "$HOST":"\$WORK/pspe/runs/$d/" "runs/$d/" && log "fetched $d"
done

# Aggregate locally with the key each runner uses.
"$PY" eval/run_seeds.py --aggregate-only --seeds 0 1 2 3 4 --out runs/constraint_fix --key arm >/dev/null 2>&1 \
    && log "aggregated constraint_fix" || log "constraint_fix aggregation failed"
"$PY" eval/run_seeds.py --aggregate-only --seeds 0 1 2 --out runs/explain_real_v2 --key arm >/dev/null 2>&1 \
    && log "aggregated explain_real_v2" || log "explain_real_v2 aggregation failed"

"$PY" - <<'PYEOF'
import json, glob, numpy as np
per = {}
for f in sorted(glob.glob("runs/resolution_seeds/seed_*/dar_resolution.json")):
    for g, rec in json.load(open(f)).items():
        per.setdefault(int(g), []).append(float(rec["rel_l2_final"]))
rows = {g: {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v, ddof=1)) if len(v) > 1 else None,
            "n": len(v), "nan": int(np.isnan(v).sum())} for g, v in sorted(per.items())}
json.dump(rows, open("runs/resolution_seeds/resolution_seeds.json", "w"), indent=2)
print("resolution:", json.dumps(rows))
PYEOF

{
    echo "# Vista batch — fetched $(date -u '+%Y-%m-%d %H:%M:%SZ')"
    echo; echo "## constraint_fix"; cat runs/constraint_fix/results_seeds.md 2>/dev/null
    echo; echo "## explain_real_v2"; cat runs/explain_real_v2/results_seeds.md 2>/dev/null
    echo; echo "## resolution_seeds"; cat runs/resolution_seeds/resolution_seeds.json 2>/dev/null
} > runs/VISTA_BATCH_RESULTS.md
log "summary written to runs/VISTA_BATCH_RESULTS.md"
