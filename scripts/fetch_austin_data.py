"""The Austin data stack: terrain, buildings, floodplain, crossings, gauges.

Austin is a better subject for this work than the Richmond valley was, and not
for any reason to do with the model. It is that every layer the portal has had
to approximate is published here, open, keyless and queryable by bounding box.

| what                | Richmond valley                          | Austin |
|---------------------|------------------------------------------|--------|
| terrain             | Copernicus GLO-30, a SURFACE model        | USGS 3DEP bare earth at 1 m |
| building footprints | OpenStreetMap, no heights, no addresses   | 585,916 city footprints with ground AND roof elevation |
| addresses           | NSW service, joined at 85.9%              | 465,977 city address points |
| design flood levels | council PDFs, per address                 | FEMA base flood elevations, plus Austin's own fully-developed floodplain |
| river forcing       | bankfull guessed from a 30 m DEM, an order of magnitude under the flood of record | 34 live USGS gauges |
| roads that flood    | inferred from solved depth                | 22 instrumented low water crossings |

The terrain line is the one that matters most. Copernicus is a Digital SURFACE
Model and reads a median 2.99 m high against USGS bare earth over downtown
Austin, high at 87.9% of cells (section 5.14f). 3DEP is bare earth, free, and
arrives already projected onto the grid the solver asks for; `build_domain`
takes `source="auto"` and uses it inside the United States.

The gauge line is the one that changes what the interface can ask. At Lismore
the trip view had to offer a text box for discharge, because bankfull derived
from a 30 m DEM came out an order of magnitude under the 2022 flood of record
and there was no gauge feed to replace it. Austin has 34 real-time gauges
inside one box, so the question "how much water is coming down" has an answer
that is measured rather than typed.

Nothing here needs a key or an account. Licences: 3DEP and USGS gauge data are
public domain as US Government works; City of Austin open data is published
under the city's open data terms; FEMA NFHL is public.

Usage:
    python3 scripts/fetch_austin_data.py --out data/austin
    python3 scripts/fetch_austin_data.py --out data/austin --bbox -97.80,30.12,-97.70,30.20
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

AUSTIN = "https://services.arcgis.com/0L95CJ0VTaxqcmED/arcgis/rest/services"
FEMA = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer"
NWIS = "https://waterservices.usgs.gov/nwis"
UA = {"User-Agent": "PSPE-floodplain-planner/1.0"}
FT = 0.3048

# City layers worth having, and what each one answers.
LAYERS = {
    "buildings": (f"{AUSTIN}/STRUCTURE_building_footprints_2013/FeatureServer/0",
                  "OBJECTID,MAX_HEIGHT,ELEVATION,BASE_ELEVATION,FEATURE"),
    "addresses": (f"{AUSTIN}/LOCATION_address_points/FeatureServer/0",
                  "OBJECTID,ADDRESS,FULL_STREET_NAME,ADDRESS_TYPE"),
    "low_water_crossings": (f"{AUSTIN}/INLANDWATERS_flood_low_water_crossings/FeatureServer/0",
                            "OBJECTID,GAGE_NUMBER,CROSSING_DESCRIPTION,CROSSING_TYPE"),
    "floodplain_austin": (f"{AUSTIN}/INLANDWATERS_austin_full_develop_floodplain/FeatureServer/0",
                          "OBJECTID,FLOOD_ZONE,FLOODWAY,SOURCE_CITATION"),
    "floodplain_fema": (f"{AUSTIN}/INLANDWATERS_greater_austin_fema_floodplain/FeatureServer/0",
                        "OBJECTID,FLOOD_ZONE,FLOODWAY,FIRM_PANEL,EFFECTIVE_DATE"),
    # FEMA's own service, for the published 1% design water surface elevation.
    "base_flood_elevations": (f"{FEMA}/16", "*"),
}


def _get(url: str, params: dict, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}", headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def fetch_layer(url: str, fields: str, bbox: tuple[float, float, float, float],
                page: int = 1000) -> list[dict]:
    """Every feature in the box, paged past the server's record cap."""
    w, s, e, n = bbox
    out: list[dict] = []
    offset = 0
    while True:
        d = _get(f"{url}/query", {
            "f": "geojson", "where": "1=1", "outFields": fields,
            "geometry": f"{w},{s},{e},{n}", "geometryType": "esriGeometryEnvelope",
            "inSR": 4326, "outSR": 4326, "spatialRel": "esriSpatialRelIntersects",
            "returnGeometry": "true", "resultOffset": offset, "resultRecordCount": page,
        })
        got = d.get("features", [])
        out += got
        if len(got) < page:
            return out
        offset += page


def clean_buildings(feats: list[dict]) -> list[dict]:
    """Feet to metres, and the -9999999999 sentinel out.

    The city publishes BASE_ELEVATION (ground at the structure) and ELEVATION
    (its top) in US survey feet, from 2012 lidar, with missing values coded as
    a large negative rather than null -- so a mean taken straight off the
    service comes back as -70,005,326 and looks like a units problem when it is
    a sentinel problem.
    """
    kept = 0
    for f in feats:
        a = f["properties"]
        for src, dst in (("BASE_ELEVATION", "ground_m"), ("ELEVATION", "top_m"),
                         ("MAX_HEIGHT", "height_m")):
            v = a.get(src)
            a[dst] = round(v * FT, 2) if v is not None and v > -1e6 else None
        kept += a["ground_m"] is not None
    return feats, kept                                   # noqa: B901


def fetch_gauges(bbox: tuple[float, float, float, float]) -> dict:
    """Active real-time stream gauges in the box, with their latest readings."""
    w, s, e, n = bbox
    req = urllib.request.Request(
        f"{NWIS}/site/?format=rdb&bBox={w},{s},{e},{n}"
        "&siteType=ST&siteStatus=active&hasDataTypeCd=iv", headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        rdb = r.read().decode()
    sites = []
    for line in rdb.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split("\t")
        if len(p) > 6 and p[1].isdigit():
            sites.append({"site": p[1], "name": p[2],
                          "lat": float(p[4]), "lon": float(p[5])})
    if not sites:
        return {"sites": [], "latest": []}
    ids = ",".join(x["site"] for x in sites)
    d = _get(f"{NWIS}/iv/", {"format": "json", "sites": ids,
                             "parameterCd": "00060,00065", "siteStatus": "active"})
    latest = []
    for t in d.get("value", {}).get("timeSeries", []):
        vals = t["values"][0]["value"]
        if not vals:
            continue
        latest.append({
            "site": t["sourceInfo"]["siteCode"][0]["value"],
            "name": t["sourceInfo"]["siteName"],
            "variable": t["variable"]["variableName"].split(",")[0],
            "unit": t["variable"]["unit"]["unitCode"],
            "value": vals[-1]["value"], "at": vals[-1]["dateTime"],
        })
    return {"sites": sites, "latest": latest}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/austin")
    ap.add_argument("--bbox", default="-97.80,30.18,-97.68,30.30",
                    help="west,south,east,north (default: central Austin)")
    ap.add_argument("--skip", default="", help="comma-separated layer names to skip")
    args = ap.parse_args()

    bbox = tuple(float(x) for x in args.bbox.split(","))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    summary: dict = {"bbox": list(bbox), "layers": {}}

    for key, (url, fields) in LAYERS.items():
        if key in skip:
            continue
        try:
            feats = fetch_layer(url, fields, bbox)
        except Exception as exc:                          # noqa: BLE001
            print(f"  {key:24s} FAILED: {type(exc).__name__}: {exc}")
            summary["layers"][key] = {"error": str(exc)[:200]}
            continue
        note = ""
        if key == "buildings":
            feats, with_elev = clean_buildings(feats)
            note = f", {with_elev} with a ground elevation"
            summary["layers"][key] = {"n": len(feats), "with_ground_elevation": with_elev}
        else:
            summary["layers"][key] = {"n": len(feats)}
        (out / f"{key}.geojson").write_text(json.dumps(
            {"type": "FeatureCollection", "features": feats}))
        print(f"  {key:24s} {len(feats):7d} features{note}")

    g = fetch_gauges(bbox)
    (out / "gauges.json").write_text(json.dumps(g, indent=2) + "\n")
    summary["layers"]["gauges"] = {"n": len(g["sites"]), "readings": len(g["latest"])}
    print(f"  {'gauges':24s} {len(g['sites']):7d} active, {len(g['latest'])} live readings")

    summary["sources"] = {
        "terrain": "USGS 3DEP bare earth, via pspe.simulate.real.anywhere "
                   "build_domain(source='3dep'); public domain",
        "buildings/addresses/crossings/floodplain": "City of Austin open data, "
                                                    "ArcGIS FeatureServer",
        "base_flood_elevations": "FEMA National Flood Hazard Layer",
        "gauges": "USGS NWIS instantaneous values; public domain",
        "roads": "OpenStreetMap via Overpass, ODbL, fetched at request time",
    }
    summary["units"] = ("City of Austin elevations are US survey feet and are "
                        "converted to metres here; 3DEP and USGS are metres. "
                        "City and FEMA elevations are NAVD88.")
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
