#!/bin/bash
# Keep the Vista ControlMaster alive by generating real session activity.
#
#   nohup bash scripts/tacc/keepalive.sh > /tmp/vista-keepalive.log 2>&1 &
#
# ServerAliveInterval was not enough: it emits TCP-level keepalives, which the
# login node's idle-session reaper does not count as activity. Running an actual
# command through the master every few minutes does count, because it opens a
# channel and executes something. Three masters died mid-job before this existed.
#
# Exits by itself once the master is gone, so it never becomes a stray process
# reconnecting to a host you have finished with.
set -uo pipefail
INTERVAL="${INTERVAL:-240}"
while true; do
    if ! ssh -o BatchMode=yes -o ConnectTimeout=10 vista true 2>/dev/null; then
        echo "[$(date -u '+%H:%M:%SZ')] master gone - stopping keepalive"
        exit 0
    fi
    echo "[$(date -u '+%H:%M:%SZ')] master alive"
    sleep "$INTERVAL"
done
