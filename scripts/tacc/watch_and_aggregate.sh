#!/bin/bash
# Wait for this user's jobs, then aggregate whatever they produced.
#
# Split out from driver.sh so a watcher can be attached to jobs that are
# *already* running (the driver only waits on ids it submitted itself).
#
#   tmux new -d -s pspewatch 'bash scripts/tacc/watch_and_aggregate.sh'
set -uo pipefail
cd "$WORK/pspe"
LOG="$WORK/pspe/driver.log"
export VENV="$WORK/pspe-venv" REPO="$WORK/pspe"
log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%SZ')] $*" | tee -a "$LOG" >&2; }

log "watcher: waiting for all jobs"
while [ "$(squeue -u "$USER" -h 2>/dev/null | wc -l)" -gt 0 ]; do
    bash scripts/tacc/write_status.sh 2>/dev/null
    sleep 60
done
log "watcher: queue empty, aggregating"

module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
source "$VENV/bin/activate"

# Each sweep is aggregated with the key its runner writes: the ablation runner
# labels rows "run", the baseline runners label them "arm".
aggregate() {  # aggregate <dir> <seeds> <key> [results-file]
    local dir="$1" seeds="$2" key="$3" file="${4:-results.json}"
    if [ ! -d "runs/$dir" ]; then log "skip $dir (absent)"; return; fi
    log "aggregating runs/$dir (key=$key, file=$file)"
    python eval/run_seeds.py --aggregate-only --seeds $seeds \
        --out "runs/$dir" --key "$key" --results-file "$file" >> "$LOG" 2>&1 \
        || log "aggregation FAILED for $dir"
}

aggregate perception_real   "0 1 2"       arm
aggregate explain_real      "0 1 2"       arm
aggregate explain_real_v2   "0 1 2"       arm
aggregate constraint_fix    "0 1 2 3 4"   arm
aggregate transfer_planning "0 1 2 3 4"   run transfer_matrix.json
# Phase 1: theory-code gap closers
aggregate conformal_real    "0 1 2"       delta
aggregate faith_weights     "0 1 2"       arm
aggregate lipschitz         "0 1 2 3 4"   arm
aggregate alpha_rule        "0 1 2 3 4"   arm
aggregate joint             "0 1 2 3 4"   arm
# Phase 3: the same two comparisons on the other testbeds
for tb in swe rdf; do
    aggregate "constraint_fix_$tb" "0 1 2 3 4" arm
    aggregate "joint_$tb"          "0 1 2 3 4" arm
done
aggregate constraint_fix_sat     "0 1 2 3 4" arm
aggregate constraint_fix_sat_rdf "0 1 2 3 4" arm

# run_resolution.py writes {grid: {...}}, not a row list, so it needs its own
# collation rather than the shared aggregator.
python - <<'PY' >> "$LOG" 2>&1
import json, pathlib
import numpy as np
root = pathlib.Path("runs/resolution_seeds")
per_grid = {}
for f in sorted(root.glob("seed_*/dar_resolution.json")):
    for grid, rec in json.loads(f.read_text()).items():
        per_grid.setdefault(grid, []).append(float(rec["rel_l2_final"]))
if per_grid:
    rows = {g: {"mean": float(np.mean(v)),
                "std": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                "n": len(v), "nan": int(np.isnan(v).sum())}
            for g, v in sorted(per_grid.items(), key=lambda kv: int(kv[0]))}
    (root / "resolution_seeds.json").write_text(json.dumps(rows, indent=2))
    print("resolution across seeds:", json.dumps(rows, indent=2))
else:
    print("no resolution results found")
PY

bash scripts/tacc/write_status.sh
log "watcher: done"
