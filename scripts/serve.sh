#!/bin/sh
# Start the flood service, or say it is already up.
#
# The portal is a client; nothing in it works without this. It is deliberately
# idempotent: running it twice is safe, because the usual reason to run it is
# that you are not sure whether it is up.
#
# No --reload by default. A reload drops the in-memory caches and, worse, can
# land in the middle of a two-minute solve; pass --reload yourself when you are
# editing Python and not waiting on a result.
set -e
cd "$(dirname "$0")/.."
PORT="${PORT:-8000}"
PY="${PYTHON:-/usr/bin/python3}"

if curl -fsS -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "already running on ${PORT}: $(curl -fsS -m 3 http://127.0.0.1:${PORT}/health)"
  exit 0
fi

mkdir -p runs
# nohup plus a closed stdin, so it outlives the shell that started it. macOS has
# no setsid, which is the obvious thing to reach for and is not there.
nohup "$PY" -m uvicorn service.app:app --port "$PORT" \
  >> runs/service.log 2>&1 < /dev/null &
echo "starting (pid $!), logging to runs/service.log"

i=0
while [ "$i" -lt 40 ]; do
  if curl -fsS -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "up on ${PORT}: $(curl -fsS -m 3 http://127.0.0.1:${PORT}/health)"
    exit 0
  fi
  sleep 1
  i=$((i + 1))
done
echo "did not come up within 40s; last lines of runs/service.log:" >&2
tail -20 runs/service.log >&2
exit 1
