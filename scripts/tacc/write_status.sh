#!/bin/bash
# Rewrite STATUS.md — read it with: ssh vista 'bash $WORK/pspe/scripts/tacc/status.sh'
cd "$WORK/pspe" || exit 1
{
    echo "# PSPE cluster status"
    echo; echo "updated: $(date -u '+%Y-%m-%d %H:%M:%SZ')"
    echo; echo '## queue'; echo '```'
    squeue -u "$USER" -o '%.10i %.14j %.9T %.8M %R' 2>/dev/null || echo '(squeue failed)'
    echo '```'
    echo; echo '## driver log (last 20)'; echo '```'; tail -20 driver.log 2>/dev/null; echo '```'
    echo; echo '## results present'; echo '```'
    for d in pdebench perception_real explain_real transfer_planning resolution_seeds; do
        [ -e "runs/$d" ] && echo "runs/$d: $(ls runs/$d 2>/dev/null | tr '\n' ' ')"
    done
    echo '```'
} > STATUS.md
