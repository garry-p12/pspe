"""The four refusals that keep the observed path honest.

Every capability in §7 of `docs/PORTAL_GUIDE.md` exists to let reality set the
model's boundary or grade its answer. Each one is only worth having if it
REFUSES when the evidence is not there, because the alternative is not a
missing number -- it is a confident number from the wrong evidence, under a
badge that says a satellite or an instrument recorded it.

Three bugs during the building of this feature were all of that species: the
wrong creek's gauge, the first gauge that happened to report, and a rainfall
fallback that ran before the observed branch. None of them raised anything.
They each produced a plausible flood.

These tests use recorded or synthetic inputs and touch no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi")

from service.app import (                                    # noqa: E402
    RoadsRequest, _reach_words, observed_discharge,
)


# --------------------------------------------------------------- reach names

def test_reach_words_do_not_match_a_different_river_in_the_same_city():
    """"Onion Creek, Austin, Texas" must not match "Colorado Rv at Austin".

    The city and the state appear in every station name in the box, so a
    naive word match makes every gauge in Austin a match for every study in
    Austin -- which is how the Colorado River, two orders of magnitude
    larger, came to be the headline for a creek.
    """
    want = _reach_words("Onion Creek, Austin, Texas")
    assert "onion" in want
    for dropped in ("creek", "austin", "texas"):
        assert dropped not in want, f"{dropped!r} would match the wrong gauge"

    def matches(name: str) -> bool:
        return any(w in name.lower() for w in want)

    assert matches("Onion Ck at US Hwy 183, Austin, TX")
    assert matches("Onion Ck at Twin Creeks Rd nr Manchaca, TX")
    assert not matches("Colorado Rv at Austin, TX")
    assert not matches("Williamson Ck at Jimmy Clay Rd, Austin, TX")


# ---------------------------------------------------- picking the right gauge

def _feed(*gauges):
    return {"supported": True, "gauges": list(gauges), "cadence_min": 15}


def _g(site, name, q, flood_levels=None):
    return {"site": site, "name": name, "q_m3s": q, "at": "2015-10-30T14:49",
            "stage_m": 4.3, "flood_levels_m": flood_levels}


BOX = dict(west=-97.82, south=30.11, east=-97.69, north=30.21)


def test_observed_discharge_refuses_rather_than_taking_another_river(monkeypatch):
    """An outage on this creek must not hand the boundary to a neighbour.

    The first version took the first gauge reporting discharge. With Onion
    Creek's own instrument down, that is Williamson Creek -- a different
    catchment, reported as an observation of this one.
    """
    import service.app as app

    monkeypatch.setattr(app, "gauges", lambda req: _feed(
        _g("08158840", "Williamson Ck at Jimmy Clay Rd, Austin, TX", 12.0),
        _g("08158000", "Colorado Rv at Austin, TX", 480.0, [4.5, 6.0, 7.5]),
    ))
    assert observed_discharge(**BOX, reach="Onion Creek, Austin, Texas") is None


def test_observed_discharge_takes_this_reach_over_a_bigger_neighbour(monkeypatch):
    """A gauge on the reach beats a larger, better-instrumented one off it."""
    import service.app as app

    monkeypatch.setattr(app, "gauges", lambda req: _feed(
        _g("08158000", "Colorado Rv at Austin, TX", 480.0, [4.5, 6.0, 7.5]),
        _g("08159000", "Onion Ck at US Hwy 183, Austin, TX", 3454.7, [4.5]),
    ))
    got = observed_discharge(**BOX, reach="Onion Creek, Austin, Texas")
    assert got is not None
    assert got["site"] == "08159000"
    assert got["on_reach"] is True


def test_observed_discharge_refuses_when_nothing_reports_discharge(monkeypatch):
    """Stage without discharge is not a discharge. It must not be invented."""
    import service.app as app

    monkeypatch.setattr(app, "gauges", lambda req: _feed(
        _g("08159000", "Onion Ck at US Hwy 183, Austin, TX", None, [4.5]),
    ))
    assert observed_discharge(**BOX, reach="Onion Creek, Austin, Texas") is None


def test_observed_discharge_prefers_the_gauge_above_the_inflow_edge(monkeypatch):
    """A boundary condition has a location, and a name match does not check it.

    Both of these are on Onion Creek and both report discharge. US Hwy 183
    sits 0.13 km PAST the box's east edge -- the outlet -- and carries NWS
    flood categories; Twin Creeks Rd sits 0.11 km before the west edge, where
    the water comes in, and carries none. Preferring flood categories picked
    the outlet, so the model was fed its own outflow at its inlet. On
    30 Oct 2015 that is 1,101.5 m3/s standing in for 342.6 m3/s, with 69% of
    the flood generated inside the box and routed a box-length too far.
    """
    import service.app as app

    us183 = _g("08159000", "Onion Ck at US Hwy 183, Austin, TX", 1101.5, [4.5])
    us183.update(lat=30.17798987, lon=-97.6886138)
    twin = _g("08158827", "Onion Ck at Twin Creeks Rd nr Manchaca, TX", 342.6)
    twin.update(lat=30.12632345, lon=-97.8211161)
    monkeypatch.setattr(app, "gauges", lambda req: _feed(us183, twin))

    # Water enters by the west edge on this reach.
    got = app.observed_discharge(**BOX, reach="Onion Creek, Austin, Texas",
                                 edge="west")
    assert got["site"] == "08158827", (
        "picked the gauge past the outlet over the one above the inlet")
    assert got["position"] == "upstream"

    # And the defect is still reachable, reported, when nothing sits upstream.
    monkeypatch.setattr(app, "gauges", lambda req: _feed(us183))
    only = app.observed_discharge(**BOX, reach="Onion Creek, Austin, Texas",
                                  edge="west")
    assert only["site"] == "08159000"
    assert only["position"] == "downstream", (
        "a reading from below the domain must say so, not pass as an inflow")


def test_observed_forcing_refuses_instead_of_becoming_a_rainstorm(monkeypatch):
    """The fall-through is a scenario, and a scenario is not an observation.

    The first fix for this moved the rainfall guard so it could not pre-empt
    the observed branch. It did not stop the observed branch RETURNING to
    that guard when the gauge had nothing to give: `river_q_observed=True`
    with a silent instrument fell through the forecast product and out into
    "the storm is applied as rainfall instead", and the caller got a
    confident flood driven by something they did not ask for.
    """
    import numpy as np

    import service.app as app

    monkeypatch.setattr(app, "observed_discharge", lambda *a, **k: None)
    monkeypatch.setattr(app, "river_discharge", lambda lat, lon: None)

    class _Dom:
        def __init__(self):
            self.z = np.full((40, 40), 100.0, dtype="float32")
            self.z[20, :] = 90.0                      # a channel across it
            self.bounds = dict(BOX)
            self.dx = 60.0

    req = app.AnalyseRequest(**BOX, name="Onion Creek", driver="river",
                             river_q_observed=True, dx=60)
    with pytest.raises(app.ObservationUnavailable, match="Nothing was substituted"):
        app.build_forcing(_Dom(), req)

    # And without the flag the same box still falls back, as it should: the
    # refusal is a promise to the caller who asked for an instrument, not a
    # new failure mode for everyone else.
    plain = app.AnalyseRequest(**BOX, name="Onion Creek", driver="river", dx=60)
    assert app.build_forcing(_Dom(), plain)["driver"] == "rain"


# ------------------------------------------------------ the replay gap refusal

def _series(start_h, n, step_min=15, q=50.0):
    from datetime import datetime, timedelta
    t0 = datetime(2013, 10, 30) + timedelta(hours=start_h)
    return [{"at": (t0 + timedelta(minutes=step_min * i)).isoformat(),
             "q_m3s": q} for i in range(n)]


def test_replay_refuses_across_a_hole_in_the_record(monkeypatch):
    """Gauges fail in the floods worth replaying.

    Onion Creek's own gauge stopped for twenty-four hours across the peak of
    31 October 2013. Interpolating that hole produces a smooth rise through
    the exact hours the flood happened, and a model that calmly reports
    nothing much occurred -- the first version returned 0.01 km2 for the
    event the neighbourhood is named for.
    """
    import service.app as app

    broken = _series(0, 20) + _series(24, 20)          # a 19 h hole
    monkeypatch.setattr(app, "discharge_series", lambda s, a, b: broken)
    monkeypatch.setattr(app, "observed_discharge",
                        lambda *a, **k: {"site": "08159000", "name": "Onion Ck",
                                         "q_m3s": 50.0, "at": None,
                                         "stage_m": None, "on_reach": True})

    out = app.replay(app.ReplayRequest(
        **BOX, name="Onion Creek", start="2013-10-30T00:00",
        end="2013-11-01T00:00"))

    assert out["ok"] is False
    assert out["stage"] == "record"
    assert out["gaps"] and out["gaps"][0]["hours"] > 2.0
    assert "hole" in out["reason"]


def test_replay_runs_when_the_record_is_whole(monkeypatch):
    """The guard must not refuse a record that is merely long."""
    import service.app as app

    whole = _series(0, 192)                            # 48 h at 15 min, no gap
    monkeypatch.setattr(app, "discharge_series", lambda s, a, b: whole)
    monkeypatch.setattr(app, "observed_discharge",
                        lambda *a, **k: {"site": "08159000", "name": "Onion Ck",
                                         "q_m3s": 50.0, "at": None,
                                         "stage_m": None, "on_reach": True})
    seen = {}

    def _fake_analyse(req):
        seen["hydrograph"] = req.hydrograph
        seen["driver"] = req.driver
        seen["run_hours"] = req.run_hours
        return {"ok": True, "flooded_km2": 1.0}

    monkeypatch.setattr(app, "analyse", _fake_analyse)
    out = app.replay(app.ReplayRequest(
        **BOX, name="Onion Creek", start="2015-10-30T00:00",
        end="2015-11-01T00:00"))

    assert out["ok"] is True
    assert out["replay"]["readings"] == 192
    assert seen["driver"] == "river"
    assert seen["hydrograph"] is whole
    # The run has to cover the hydrograph or the flood is cut off part-way up
    # its own rising limb.
    assert seen["run_hours"] >= out["replay"]["span_hours"]


# ------------------------------------------------- the separability refusal

def test_trough_rejects_a_histogram_with_no_split():
    """Otsu cannot return "nothing here", so its caller has to.

    Given a unimodal histogram -- a dry scene -- Otsu splits the land
    distribution down the middle and calls half of it water. That is how
    71 km2 of a 138 km2 box came to be reported as permanent water, under the
    badge that means a satellite recorded it.
    """
    from pspe.observe.sar import MIN_TROUGH, otsu, trough_depth

    rng = np.random.default_rng(0)
    for name, dry in (("gaussian", rng.normal(-8.0, 2.0, 40_000)),
                      ("wide gaussian", rng.normal(-8.0, 3.5, 40_000)),
                      ("left-skewed", -np.abs(rng.gamma(3, 1.4, 40_000)) - 4.0)):
        thr, _ = otsu(dry)
        assert trough_depth(dry, thr) < MIN_TROUGH, (
            f"a unimodal {name} scene would be let through")


def test_trough_accepts_a_flood_that_covers_little_of_the_box():
    """The guard must not reject a real flood for being small.

    This is the case the old separability floor got backwards. Otsu's eta is
    driven by class BALANCE: an even water/land split scores 0.91 and a
    flood covering 3% of the box scores 0.355 -- below the 0.55 floor that
    was supposed to admit it. Onion Creek in flood is about 2.5% of this
    study box, so that floor was set to reject the case it exists to
    measure.
    """
    from pspe.observe.sar import MIN_SEPARABILITY, MIN_TROUGH, otsu, trough_depth

    rng = np.random.default_rng(0)
    small = np.concatenate([rng.normal(-18.0, 1.5, 1_200),    # 3% water
                            rng.normal(-7.0, 1.5, 38_800)])
    thr, eta = otsu(small)
    assert -20.0 < thr < -10.0, "Otsu should still find the water mode"
    assert trough_depth(small, thr) > MIN_TROUGH, "a real flood was refused"
    # The regression this replaced, asserted so it cannot come back.
    assert eta < MIN_SEPARABILITY, (
        "if eta now passes for a 3% flood the old floor was not the bug "
        "this test documents")


def test_trough_accepts_an_even_split():
    from pspe.observe.sar import MIN_TROUGH, otsu, trough_depth

    rng = np.random.default_rng(0)
    wet = np.concatenate([rng.normal(-18.0, 1.5, 20_000),
                          rng.normal(-7.0, 1.5, 20_000)])
    thr, _ = otsu(wet)
    assert -20.0 < thr < -10.0
    assert trough_depth(wet, thr) > MIN_TROUGH


def test_detect_refuses_a_scene_with_no_water_land_split(monkeypatch):
    """The guard is wired into `detect`, not merely available to it."""
    import pspe.observe.sar as sar

    rng = np.random.default_rng(1)

    def _fake_read(scene, bbox, band="vv", out_shape=None):
        lin = 10.0 ** (rng.normal(-8.0, 2.0, (64, 64)) / 10.0)
        return lin, (None, None)

    monkeypatch.setattr(sar, "read_bbox", _fake_read)
    scene = sar.Scene(id="s", datetime="2026-10-05T00:34:37Z",
                      orbit_state="ascending", relative_orbit=107, assets={})
    base = sar.Scene(id="b", datetime="2026-09-23T00:34:37Z",
                     orbit_state="ascending", relative_orbit=107, assets={})

    with pytest.raises(ValueError, match="no water/land split"):
        sar.detect(scene, [base], [-97.82, 30.11, -97.69, 30.21])
