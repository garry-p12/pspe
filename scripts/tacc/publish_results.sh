#!/bin/bash
# Publish aggregated results from the cluster to GitHub, so reading them needs
# no SSH connection to TACC at all.
#
#   bash scripts/tacc/publish_results.sh
#
# The problem this solves: results live on Vista, and every route to them runs
# through an SSH session that MFA gates and the login-node reaper kills. Pushing
# the *aggregates* (kilobytes of .md/.json, not the gigabytes under runs/) to a
# branch means they can be read with `git fetch` from anywhere, whether or not a
# TACC login is currently possible.
#
# Requires push credentials on the cluster - see scripts/tacc/README.md.
set -euo pipefail
cd "$WORK/pspe"

BRANCH="${BRANCH:-vista-results}"
DEST="published_results"
mkdir -p "$DEST"

# Only aggregates. runs/ is gitignored for good reason: it is ~100 MB of
# checkpoints and per-step metrics that nobody reads out of git.
for d in seeds_v3 seeds_rest transfer_seeds perception_seeds explain_seeds \
         perception_real explain_real transfer_planning resolution_seeds pdebench; do
    for f in "runs/$d"/results_seeds.md "runs/$d"/results_seeds.json \
             "runs/$d"/results.md "runs/$d"/pdebench_results.md \
             "runs/$d"/pdebench_results.json "runs/$d"/resolution_seeds.json; do
        [ -f "$f" ] || continue
        mkdir -p "$DEST/$d"
        cp "$f" "$DEST/$d/"
    done
done
cp STATUS.md "$DEST/" 2>/dev/null || true
cp driver.log "$DEST/" 2>/dev/null || true

git add -A "$DEST"
if git diff --cached --quiet; then
    echo "no result changes to publish"
    exit 0
fi
git commit -m "Vista results: $(date -u '+%Y-%m-%d %H:%M')

Aggregated sweep outputs published from \$WORK/pspe on TACC Vista.
Raw runs/ stays gitignored; this is the summary layer only."
git push origin "HEAD:$BRANCH"
echo "published to origin/$BRANCH"
