#!/usr/bin/env bash
# Stage everything an outside slide-building tool needs, in one folder, with
# filenames that carry the slide order.
#
# A tool handed `deck_polder.png` and `16-street-solved.jpg` has no way to know
# which slide either belongs to, and will guess. Numbering the files removes
# the guess.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/docs/deck_bundle"
FIG="$ROOT/docs/figures/deck"
SHOT="$ROOT/docs/images/portal-guide"

rm -rf "$OUT"; mkdir -p "$OUT/images"

cp "$ROOT/docs/DECK_POINTERS.md" "$OUT/OUTLINE.md"

# slide number -> source file. Slides with no image are absent by design.
copy() { cp "$2" "$OUT/images/$1"; }

copy "slide-01-title-portal-overview.jpg"        "$SHOT/01-forecast-open.jpg"
copy "slide-03-street-houses-flooded.jpg"        "$SHOT/16-street-solved.jpg"
copy "slide-04-silent-failures.png"              "$FIG/deck_measurement.png"
copy "slide-05-system-architecture.png"          "$ROOT/docs/figures/pspe_system.png"
copy "slide-06-what-it-offers.png"               "$FIG/deck_offers.png"
copy "slide-07-the-estimator.png"                "$FIG/deck_framework.png"
copy "slide-10-how-the-twin-works.png"           "$FIG/deck_twin_pipeline.png"
copy "slide-11-solved-anywhere.jpg"              "$SHOT/05-forecast-town.jpg"
copy "slide-12-elevation-data-lesson.png"        "$FIG/deck_dem.png"
copy "slide-13-roads-cut.jpg"                    "$SHOT/06-roads-open.jpg"
copy "slide-14-warning-time.png"                 "$FIG/deck_warning.png"
copy "slide-15-the-decision-gym.png"             "$FIG/deck_gym_loop.png"
copy "slide-17-operating-the-structures.png"     "$SHOT/26-operate-result-panel.png"
copy "slide-18-headline-matched-budget.png"      "$FIG/deck_matched_budget.png"
copy "slide-19-safety-violations.png"            "$FIG/deck_margin.png"
copy "slide-20-heuristic-beats-learning.png"     "$FIG/deck_polder.png"
copy "slide-21-why-our-planner-failed.png"       "$FIG/deck_collapse.png"
copy "slide-22-only-three-behaviours.png"        "$FIG/deck_flood_frontier.png"
copy "reserve-R1-null-floor.png"                 "$FIG/deck_null_floor.png"
copy "reserve-R2-mixing-rule.png"                "$FIG/deck_mixing_rule.png"

cat > "$OUT/README.txt" <<'TXT'
WHAT IS IN HERE

  OUTLINE.md   the deck: 26 slides in six acts, then reserve slides R1-R7,
               then a caption for every figure.
  images/      one file per slide that has an image. The filename carries the
               slide number from OUTLINE.md. Slides not listed here are text
               only and need no image.

HOW TO READ THE OUTLINE

  Each slide has SAY (the line to deliver), SHOW (which image), and sometimes
  IF ASKED (backup detail for questions, NOT for the slide body).

  Put SAY on the slide or in the speaker notes. Never put IF ASKED on a slide.

PNG is supplied. Vector versions (svg, pdf) of every figure are in
docs/figures/deck/ if the tool accepts them -- they will stay sharp at any
projector size.
TXT

echo "staged $(ls "$OUT/images" | wc -l | tr -d ' ') images + outline in docs/deck_bundle"
du -sh "$OUT"
