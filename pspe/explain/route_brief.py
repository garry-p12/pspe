"""The Explain stage for a route: a brief that can be checked against the plan.

The rest of this package explains a planner's *action* on a PDE: the brief is
templated and parseable so `faithfulness.py` can map the text back to an action
distribution and score whether it really describes what the planner did. A
route brief is held to the same standard, and for the same reason. A paragraph
that sounds like an explanation and quietly disagrees with the route drawn on
the map is worse than no paragraph, because the operator will read it and act
on it.

So every claim this emits is recoverable by `parse`, and `faithfulness`
compares the recovered claims against the route record itself. A brief scores
1.0 only when its departure call, duration, distance, named roads, deepest
water and deadline all match the plan. The surface form is constrained to make
that parse exact; within those constraints it is written for someone deciding
whether to drive, not for a log.

Numbers are quantised where a reader would not trust the extra digit anyway --
minutes to the minute, distance to 0.1 km, depth to the centimetre -- so the
parse is exact rather than a rounding of the plan (the same reason `brief.py`
quantises amplitudes to a 0.05 grid).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


def clock(minutes: float) -> str:
    """Minutes from the start of the run, as a duration a person reads."""
    m = int(round(minutes))
    if m < 60:
        return f"{m} minutes"
    h, r = divmod(m, 60)
    return f"{h}h{r:02d}"


def cm(depth_m: float) -> int:
    return int(round(depth_m * 100))


@dataclass
class RouteBrief:
    """Everything a route brief claims, recovered from its text."""
    decision: str = "none"            # now | later | impossible
    minutes: float | None = None
    km: float | None = None
    via: list[str] = field(default_factory=list)
    deepest_cm: int | None = None
    deepest_road: str | None = None
    deadline_min: float | None = None
    cut_roads: list[str] = field(default_factory=list)


def context_from(out: dict) -> dict:
    """Pull the claims out of a `/route` response. One source, not two.

    The brief is built from the same record the map is drawn from, so there is
    no second computation of "which roads" or "how deep" that could drift from
    the first.
    """
    r = out.get("route")
    water = (r or {}).get("water_on_the_way") or []
    deepest = max(water, key=lambda c: c["depth_m"]) if water else None
    return {
        "ok": bool(r),
        "leave_now": bool(out.get("leave_now")),
        "minutes": (r or {}).get("minutes"),
        "km": (r or {}).get("km"),
        "via": list((r or {}).get("via") or []),
        "deepest_m": deepest["depth_m"] if deepest else None,
        "deepest_road": (deepest or {}).get("name"),
        "deepest_after_min": (deepest or {}).get("minutes_after_leaving"),
        "deadline_min": out.get("deadline_min"),
        "depart_min": (r or {}).get("depart_min", 0.0),
        "direct_cuts": [c for c in (out.get("direct_road_cuts") or [])],
        "dry_minutes": ((out.get("dry_route") or {}) or {}).get("minutes"),
        "vehicle_depth_m": out.get("vehicle_depth_m"),
        # The cut length at t=0 is always about zero -- every run starts dry --
        # so a brief that denies a route must quote the flood that denied it,
        # not the state before it arrived.
        "cut_km_now": out.get("cut_km_now"),
        "cut_km_peak": out.get("cut_km_peak"),
        "network_km": out.get("network_km"),
        "static_minutes": ((out.get("static_route") or {}) or {}).get("minutes"),
    }


def render(ctx: dict) -> str:
    """The brief. Every sentence here is recovered by `parse`."""
    parts: list[str] = []

    if not ctx["ok"]:
        parts.append("No route. Every way through is under more water than "
                     f"{cm(ctx['vehicle_depth_m'] or 0.25)} cm.")
        for c in ctx["direct_cuts"][:3]:
            parts.append(f"Cut: {c['name'] or 'an unnamed road'} under "
                         f"{cm(c['depth_m'])} cm.")
        parts.append(f"Up to {ctx['cut_km_peak'] or 0:.0f} km of "
                     f"{ctx['network_km']:.0f} km is impassable.")
        return " ".join(parts)

    if ctx["leave_now"]:
        parts.append("Leave now.")
    else:
        parts.append(f"Leaving now does not get through; leave at "
                     f"{clock(ctx['depart_min'])}.")

    parts.append(f"Drive {int(round(ctx['minutes']))} minutes, "
                 f"{ctx['km']:.1f} km.")
    if ctx["via"]:
        roads = ctx["via"][:3]
        joined = roads[0] if len(roads) == 1 else \
            ", ".join(roads[:-1]) + " then " + roads[-1]
        parts.append(f"Via {joined}.")

    # The flood is the reason this is not an ordinary route, so it is named
    # rather than left as a property of a line on a map.
    if ctx["deepest_m"] and ctx["deepest_m"] > 0.05:
        where = ctx["deepest_road"] or "an unnamed road"
        parts.append(f"Deepest water {cm(ctx['deepest_m'])} cm on {where}, "
                     f"{int(round(ctx['deepest_after_min'] or 0))} minutes in.")
    else:
        parts.append("No standing water on this route.")

    for c in ctx["direct_cuts"][:2]:
        parts.append(f"Avoids {c['name'] or 'an unnamed road'}, under "
                     f"{cm(c['depth_m'])} cm.")

    if ctx["dry_minutes"] is not None:
        extra = (ctx["minutes"] or 0) - ctx["dry_minutes"]
        if extra >= 1:
            parts.append(f"The flood adds {int(round(extra))} minutes.")

    if ctx["deadline_min"] is not None:
        parts.append(f"Deadline {clock(ctx['deadline_min'])} from now.")
    else:
        parts.append("No deadline within the modelled window.")

    return " ".join(parts)


_NUM = r"(-?\d+(?:\.\d+)?)"


def parse(text: str) -> RouteBrief:
    """Recover the claims. The inverse of `render`, and nothing more."""
    b = RouteBrief()
    if re.search(r"\bNo route\b", text):
        b.decision = "impossible"
    elif re.search(r"\bLeave now\b", text):
        b.decision = "now"
    elif re.search(r"leave at\b", text):
        b.decision = "later"

    m = re.search(rf"Drive {_NUM} minutes, {_NUM} km", text)
    if m:
        b.minutes, b.km = float(m.group(1)), float(m.group(2))

    m = re.search(r"Via ([^.]+)\.", text)
    if m:
        b.via = [p.strip() for p in re.split(r",| then ", m.group(1)) if p.strip()]

    m = re.search(rf"Deepest water {_NUM} cm on ([^,]+),", text)
    if m:
        b.deepest_cm, b.deepest_road = int(float(m.group(1))), m.group(2).strip()
    elif re.search(r"No standing water on this route", text):
        b.deepest_cm = 0

    for m in re.finditer(rf"(?:Avoids|Cut:) ([^,]+?),? under {_NUM} cm", text):
        b.cut_roads.append(m.group(1).strip())

    m = re.search(rf"Deadline (?:{_NUM}h{_NUM}|{_NUM} minutes) from now", text)
    if m:
        b.deadline_min = (float(m.group(1)) * 60 + float(m.group(2))
                          if m.group(1) else float(m.group(3)))
    return b


def checked(out: dict) -> tuple[str | None, dict]:
    """Render a brief and refuse to return one that misdescribes the plan.

    The module's whole premise is that an explanation which disagrees with the
    decision is worse than no explanation, and `faithfulness` can tell the
    difference. Scoring it offline proves the renderer worked on the day it was
    written; scoring it here makes it a property of every response, the way
    `margins.conformal_quantile` returns None rather than a number it cannot
    support. A brief that does not round-trip is withheld, and the structured
    fields -- which the interface renders anyway, and which the brief is only a
    summary of -- stand on their own.
    """
    ctx = context_from(out)
    text = render(ctx)
    score, bad = faithfulness(text, ctx)
    report = {"faithfulness": round(score, 4), "mismatched": bad,
              "withheld": score < 1.0}
    return (text if score >= 1.0 else None), report


def faithfulness(text: str, ctx: dict) -> tuple[float, list[str]]:
    """Fraction of the brief's checkable claims that match the plan.

    Returns the score and the names of the claims that disagree, because a
    score on its own tells you something is wrong and not what. Only claims the
    brief actually makes are checked: a brief that stays silent about the
    deadline is incomplete, not unfaithful, and incompleteness is caught by the
    claim count rather than by the score.
    """
    b = parse(text)
    checks: list[tuple[str, bool]] = []

    want = ("impossible" if not ctx["ok"] else
            "now" if ctx["leave_now"] else "later")
    checks.append(("decision", b.decision == want))

    if not ctx["ok"]:
        # The no-route branch makes claims too, and checking only the verdict
        # let through "no way is passable" beside "0 km is impassable" -- two
        # sentences of the same brief disagreeing, scored 1.000. The weakest
        # part of a guard is wherever it stops looking.
        named = [c["name"] or "an unnamed road" for c in ctx["direct_cuts"][:3]]
        checks.append(("cut_roads", b.cut_roads == named))
        checks.append(("cut_km", f"{ctx['cut_km_peak'] or 0:.0f} km" in text))
        checks.append(("not_self_contradictory",
                       (ctx["cut_km_peak"] or 0) > 0))

    if ctx["ok"]:
        checks.append(("minutes", b.minutes is not None and
                       abs(b.minutes - round(ctx["minutes"])) < 1.0))
        checks.append(("km", b.km is not None and
                       abs(b.km - round(ctx["km"], 1)) < 0.05))
        checks.append(("via", b.via == [v for v in ctx["via"][:3]]))
        deep = 0 if not ctx["deepest_m"] or ctx["deepest_m"] <= 0.05 else cm(ctx["deepest_m"])
        checks.append(("deepest", b.deepest_cm == deep))
        if ctx["deadline_min"] is not None:
            checks.append(("deadline", b.deadline_min is not None and
                           abs(b.deadline_min - ctx["deadline_min"]) < 1.0))
        named = [c["name"] or "an unnamed road" for c in ctx["direct_cuts"][:2]]
        checks.append(("avoids", b.cut_roads == named))

    score = sum(1 for _, ok in checks if ok) / max(len(checks), 1)
    return score, [name for name, ok in checks if not ok]
