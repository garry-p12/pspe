#!/bin/bash
# Poll until none of $JOBS is PENDING or RUNNING, then print each one's state.
HOST="${HOST:-vista2}"
while true; do
  live=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" "sacct -j $JOBS -X -n -o State" 2>/dev/null | grep -cE "PENDING|RUNNING")
  [ -z "$live" ] && { sleep 120; continue; }
  [ "$live" = "0" ] && break
  sleep 120
done
ssh -o BatchMode=yes "$HOST" "sacct -j $JOBS -X -n -o JobID,JobName%14,State,Elapsed"
