#!/bin/bash
# Cluster-side work driver. Runs inside tmux on a Vista login node and owns the
# remaining experiment queue end to end.
#
#   tmux new -d -s pspe 'bash scripts/tacc/driver.sh'
#   tmux attach -t pspe          # to watch
#
# Why this exists: driving the queue over an ssh ControlMaster from a laptop
# failed three times in one session — the master socket is reaped and every
# in-flight poll dies with it. Here the login node owns the loop, so a dropped
# connection costs nothing; reconnecting is only needed to *read* status.
#
# Two layers of durability:
#   * jobs are chained with SLURM dependencies, so the scheduler runs the queue
#     even if this driver is killed;
#   * every step appends to driver.log and rewrites STATUS.md, so progress is
#     readable with one short ssh command.
set -uo pipefail

cd "$WORK/pspe"
ALLOC="${ALLOC:-ATM23014}"
LOG="$WORK/pspe/driver.log"
STATUS="$WORK/pspe/STATUS.md"
export VENV="$WORK/pspe-venv" REPO="$WORK/pspe"
export HF_HOME="$WORK/hf_cache"

# stderr, never stdout: `submit()` runs inside $(...), so anything log() writes
# to stdout is captured as part of the job id and wait_for then polls garbage.
log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%SZ')] $*" | tee -a "$LOG" >&2; }

write_status() {
    {
        echo "# PSPE cluster status"
        echo
        echo "updated: $(date -u '+%Y-%m-%d %H:%M:%SZ')"
        echo
        echo '## queue'
        echo '```'
        squeue -u "$USER" -o '%.10i %.12j %.9T %.8M %R' 2>/dev/null || echo '(squeue failed)'
        echo '```'
        echo
        echo '## driver log (last 25)'
        echo '```'
        tail -25 "$LOG"
        echo '```'
        echo
        echo '## results present'
        echo '```'
        for d in seeds_v3 perception_seeds explain_seeds transfer_planning pdebench resolution_seeds; do
            if [ -e "runs/$d" ]; then echo "runs/$d: $(ls runs/$d 2>/dev/null | tr '\n' ' ')"; fi
        done
        ls -t pspe-*.o* 2>/dev/null | head -5
        echo '```'
    } > "$STATUS"
}

submit() {  # submit <script> [extra sbatch args...]; echoes job id
    local script="$1"; shift
    local id
    id=$(sbatch -A "$ALLOC" "$@" "$script" 2>&1 | grep -oE '[0-9]+$')
    if [ -z "$id" ]; then log "SUBMIT FAILED: $script"; return 1; fi
    log "submitted $(basename "$script") as $id ${*:+(args: $*)}"
    echo "$id"
}

wait_for() {  # wait_for <job id...>
    local ids="$*"
    while true; do
        local n
        n=$(squeue -j "${ids// /,}" -h 2>/dev/null | wc -l)
        [ "$n" -eq 0 ] && break
        write_status
        sleep 60
    done
    log "finished: $ids"
    write_status
}

log "=== driver start (allocation $ALLOC) ==="
write_status

# --- 1. HF model cache. Compute nodes have no outbound network, so the real
# backbones must be fetched on the login node before any job needs them.
if [ ! -d "$HF_HOME/hub" ]; then
    log "pre-fetching HF backbones into $HF_HOME"
    module purge && module load gcc/13.2.0 cuda/12.6 python3/3.11.8
    source "$VENV/bin/activate"
    pip install -q "transformers>=4.44" "peft>=0.11" "accelerate>=0.33" pillow datasets 2>&1 | tail -2 | tee -a "$LOG"
    python - <<'PY' 2>&1 | tail -5 | tee -a "$LOG"
from huggingface_hub import snapshot_download
for repo in ("google/siglip-base-patch16-224", "Qwen/Qwen2.5-0.5B-Instruct"):
    print("fetching", repo)
    snapshot_download(repo)
print("hf cache ready")
PY
else
    log "HF cache already present, skipping fetch"
fi

# --- 2. The experiment queue. Independent jobs go in together; the scheduler
# runs them concurrently on separate nodes.
# PDEBench already produced results (job 928714); do not spend a node redoing it.
PDEB=""
if [ -f runs/pdebench/pdebench_results.json ]; then
    log "pdebench results already present, skipping"
else
    PDEB=$(submit scripts/tacc/vista_pdebench.slurm)
fi
TRANS=$(submit scripts/tacc/vista_transfer_planning.slurm)
RES=$(submit scripts/tacc/vista_resolution.slurm)
PERC=$(submit scripts/tacc/vista_perception_real.slurm)
EXPL=$(submit scripts/tacc/vista_explain_real.slurm)

wait_for $PDEB $TRANS $RES $PERC $EXPL

log "=== all jobs complete ==="
write_status
