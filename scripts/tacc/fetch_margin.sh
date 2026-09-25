#!/usr/bin/env bash
# Pull the conformal-margin results back from Vista and rebuild the figure.
#
#   scripts/tacc/fetch_margin.sh            # fetch whatever exists
#   scripts/tacc/fetch_margin.sh --wait     # block until the queue is empty first
#
# Safe to run repeatedly: rsync only moves what changed, and the figure script
# merges every results.json it finds, so partial fetches give a partial curve
# rather than an error.
set -uo pipefail
HOST="${HOST:-vista2}"
REMOTE="${REMOTE:-/work/11755/gurup12/vista/pspe}"
cd "$(dirname "$0")/../.."

if [ "${1:-}" = --wait ]; then
  echo "waiting for the queue to drain ..."
  while ssh -o ConnectTimeout=20 "$HOST" "squeue -u gurup12 -h" 2>/dev/null | grep -q .; do
    sleep 60
  done
  echo "queue empty"
fi

for d in margin_choice ndws_margin ndws_calib ndws_perturb margin_synthetic; do
  mkdir -p "runs/$d"
  # Results and logs only: the per-iteration metrics.jsonl files are large and
  # nothing downstream reads them.
  rsync -avz --include='*/' --include='results.md' --include='results.json' \
        --include='*.log' --exclude='*' \
        "$HOST:$REMOTE/runs/$d/" "runs/$d/" 2>&1 | tail -3
done

echo
echo "=== calibration checks (the wildfire runs refuse to be reported if the"
echo "=== constraint does not separate; look for WARNING here) ==="
grep -h -E "limit d =|b / headroom|VERDICT|WARNING|NOTE \[" \
     runs/ndws_margin/*.log runs/ndws_perturb/*.log 2>/dev/null | sort -u || echo "  (none yet)"

echo
# runs/ndws_margin/ holds the superseded per-fire-calibration runs, whose
# margins consumed the whole limit. They are kept for the record and not shown.
for f in runs/margin_choice*/*/results.md runs/ndws_perturb/*/results.md; do
  [ -f "$f" ] && { echo "--- $f"; cat "$f"; echo; }
done

"${PYTHON:-/opt/anaconda3/envs/pspe/bin/python}" scripts/make_margin_figures.py 2>&1 | tail -3
