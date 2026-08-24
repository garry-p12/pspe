#!/bin/bash
# One-shot status read. Cheap enough to survive a flaky connection:
#   ssh <host> 'bash $WORK/pspe/scripts/tacc/status.sh'
cat "$WORK/pspe/STATUS.md" 2>/dev/null || echo "no STATUS.md yet — driver has not started"
