#!/bin/bash
# Poll Vista; when one copy of an experiment starts (or finishes), cancel its
# other copies (same output dir). Prints a line per state change; exits when
# nothing is left in the queue.
HOST="${HOST:-vista2}"
# group -> ids ; first to reach RUNNING wins
declare -A GROUP=([1013314]=cfix_swe [1013316]=cfix_swe [1013315]=joint_rdf [1013317]=joint_rdf)
declare -A PART=([1013314]=gh [1013315]=gh [1013316]=gh-dev [1013317]=gh-dev)
ALL=$(IFS=,; echo "${!GROUP[*]}")
declare -A WINNER; prev=""
while true; do
  cur=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" "sacct -j $ALL -X -n -P -o JobID,State,Elapsed 2>/dev/null" 2>/dev/null | sort)
  [ -z "$cur" ] && { echo "poll failed (ssh)"; sleep 120; continue; }
  while IFS='|' read -r id st el; do
    st=${st%% *}; g=${GROUP[$id]:-}; [ -z "$g" ] && continue
    if [ -z "${WINNER[$g]:-}" ] && echo "$st" | grep -Eq "RUNNING|COMPLETED|FAILED|TIMEOUT|OUT_OF_MEMORY"; then
      WINNER[$g]=$id
      for o in "${!GROUP[@]}"; do
        [ "${GROUP[$o]}" = "$g" ] && [ "$o" != "$id" ] && ssh -o BatchMode=yes "$HOST" "scancel $o" 2>/dev/null && echo "CANCEL $g on ${PART[$o]} ($o): ${PART[$id]} copy ($id) is $st"
      done
    fi
  done <<< "$cur"
  states=$(echo "$cur" | cut -d"|" -f1,2)
  if [ "$states" != "$prev" ]; then
    echo "$cur" | while IFS='|' read -r id st el; do echo "  ${GROUP[$id]:-?} ${PART[$id]:-?} ($id): $st $el"; done
    prev=$states
  fi
  live=$(echo "$cur" | grep -Ec "PENDING|RUNNING|COMPLETING")
  [ "$live" -eq 0 ] && { echo "ALL JOBS TERMINAL"; exit 0; }
  sleep 120
done
