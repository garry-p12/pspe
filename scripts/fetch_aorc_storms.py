"""Hourly 1 km rainfall over Onion Creek, for the storms that actually happened.

`FloodGreenInfra-v0` needs a rain field, and the two candidate sources are not
close in quality:

* **ERA5**, via Open-Meteo's archive, is keyless and instant and gives a real
  hyetograph for any date back to 1940 -- but at 9-31 km it is one number for
  the whole study box and it understates convective peaks badly. It returns
  92 mm total for the Halloween 2013 Onion Creek flood; several hundred fell.
* **AORC v1.1** is NOAA's Analysis of Record for Calibration: hourly, ~1 km,
  CONUS, 1979 onwards, public on S3 with no credentials. It is the right
  answer and this script fetches it.

What it pulls is deliberately small. A storm window is about three days over a
0.3 deg box, so each event is a few hundred kilobytes once subset -- the
expensive part is that S3 serves whole chunks (144 hours x 128 x 256 cells),
so the read streams more than it keeps. Four storms is minutes, not hours.

A note on the variable: `APCP_surface` is accumulated precipitation in
kg/m^2 over the hour, which is millimetres. It is NOT a rate, so it is used
as mm/h directly and converted to the solver's m/s at the point of use.

    python3 scripts/fetch_aorc_storms.py --out data/aorc
    python3 scripts/fetch_aorc_storms.py --storm "Halloween 2013"

Needs `s3fs`, `xarray` and `zarr`. On TACC Vista the `dsc` conda environment
has all three; locally, install them or run this there and copy the result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: The Onion Creek study box, with a margin so the solver's edges are not the
#: raster's edges.
BOX = {"west": -97.90, "south": 30.05, "east": -97.60, "north": 30.35}

#: Storms that put water in Onion Creek. The first two are the floods Austin's
#: buyout programme followed; the dates bracket each event so the hyetograph
#: carries its own rise and recession rather than starting at the peak.
#: Each one was checked for rain before being listed. "Tax Day 2018" was in
#: this table first and came back at 3.1 mm over three days -- a real date and
#: not a storm here, which would have been a dead episode wearing an event's
#: name. These seven all exceed 45 mm over the box.
STORMS = {
    "Halloween 2013":    ("2013-10-30", "2013-11-01"),   # 135 mm, the buyout flood
    "Memorial Day 2015": ("2015-05-23", "2015-05-25"),   # 133 mm
    "October 2015":      ("2015-10-29", "2015-10-31"),   # 245 mm, the wettest here
    "May 2016":          ("2016-05-26", "2016-05-28"),   # 104 mm
    "October 2018":      ("2018-10-15", "2018-10-17"),   #  73 mm
    "May 2019":          ("2019-05-08", "2019-05-10"),   #  72 mm
    "August 2022":       ("2022-08-21", "2022-08-23"),   #  48 mm
}

#: Heat waves, for `HeatRetrofit-v0`. Same bucket, same box, different
#: variable: 2 m air temperature rather than accumulated precipitation.
HEATWAVES = {
    "August 2023":    ("2023-08-08", "2023-08-14"),
    "July 2022":      ("2022-07-08", "2022-07-14"),
    "June 2023":      ("2023-06-18", "2023-06-24"),
    "August 2011":    ("2011-08-01", "2011-08-07"),
    "July 2018":      ("2018-07-19", "2018-07-25"),
    "August 2019":    ("2019-08-09", "2019-08-15"),
}

BUCKET = "noaa-nws-aorc-v1-1-1km"
VAR = "APCP_surface"
HEAT_VAR = "TMP_2maboveground"


def fetch_heat(name: str, start: str, end: str, out_dir: Path) -> dict:
    """One heat wave as hourly 1 km air temperature over the box."""
    import numpy as np
    import pandas as pd
    import s3fs
    import xarray as xr

    t0, t1 = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23)
    fs = s3fs.S3FileSystem(anon=True)
    ds = xr.open_zarr(s3fs.S3Map(f"{BUCKET}/{t0.year}.zarr", s3=fs),
                      consolidated=True)
    ds = ds[[HEAT_VAR]].rename({"latitude": "lat", "longitude": "lon"})
    ds = ds.sel(lat=slice(BOX["south"], BOX["north"]),
                lon=slice(BOX["west"], BOX["east"]),
                time=slice(t0, t1)).load()
    c = ds[HEAT_VAR].values.astype("float32") - 273.15        # K -> degrees C
    slug = name.lower().replace(" ", "_")
    path = out_dir / f"aorc_heat_{slug}.npz"
    np.savez_compressed(
        path, temp_c=c, lat=ds.lat.values.astype("float32"),
        lon=ds.lon.values.astype("float32"),
        time=np.array([str(t) for t in ds.time.values]),
        name=name, source=f"NOAA AORC v1.1, s3://{BUCKET}/{t0.year}.zarr")
    hot = (c > 35.0).mean(axis=0) * 100
    return {"heatwave": name, "file": path.name, "grid": list(c.shape),
            "peak_c": round(float(c.max()), 1),
            "mean_c": round(float(c.mean()), 1),
            "hours_over_35_min_pct": round(float(hot.min()), 1),
            "hours_over_35_max_pct": round(float(hot.max()), 1)}


def fetch(name: str, start: str, end: str, out_dir: Path) -> dict:
    import numpy as np
    import pandas as pd
    import s3fs
    import xarray as xr

    t0, t1 = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23)
    if t0.year != t1.year:
        # AORC is one Zarr store per year; a storm that crosses New Year would
        # need both opened and concatenated. None of ours does, and failing
        # loudly beats silently returning half a storm.
        raise ValueError(f"{name} crosses a year boundary; not handled")

    fs = s3fs.S3FileSystem(anon=True)
    ds = xr.open_zarr(s3fs.S3Map(f"{BUCKET}/{t0.year}.zarr", s3=fs),
                      consolidated=True)
    ds = ds[[VAR]].rename({"latitude": "lat", "longitude": "lon"})
    ds = ds.sel(lat=slice(BOX["south"], BOX["north"]),
                lon=slice(BOX["west"], BOX["east"]),
                time=slice(t0, t1)).load()

    mm = ds[VAR].values.astype("float32")              # (T, lat, lon), mm/hour
    lat = ds.lat.values.astype("float32")
    lon = ds.lon.values.astype("float32")
    slug = name.lower().replace(" ", "_")
    path = out_dir / f"aorc_{slug}.npz"
    np.savez_compressed(
        path, precip_mm_h=mm, lat=lat, lon=lon,
        time=np.array([str(t) for t in ds.time.values]),
        name=name, source=f"NOAA AORC v1.1, s3://{BUCKET}/{t0.year}.zarr",
    )
    # Totals over the box, so the fetch reports something checkable.
    per_hour = mm.mean(axis=(1, 2))
    return {"storm": name, "file": path.name, "grid": list(mm.shape),
            "total_mm_mean": round(float(mm.sum(axis=0).mean()), 1),
            "total_mm_max": round(float(mm.sum(axis=0).max()), 1),
            "peak_hour_mm": round(float(per_hour.max()), 1),
            "peak_cell_mm_h": round(float(mm.max()), 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/aorc")
    ap.add_argument("--storm", action="append", default=[],
                    help="one of: " + "; ".join(STORMS))
    ap.add_argument("--heat", action="store_true",
                    help="fetch the heat waves (temperature) instead of the "
                         "storms (precipitation)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.heat:
        report = []
        for name, (a, b) in HEATWAVES.items():
            print(f"[aorc] {name} {a}..{b}", flush=True)
            r = fetch_heat(name, a, b, out)
            report.append(r)
            print(f"       {r['grid']}  peak {r['peak_c']} C, mean {r['mean_c']} C, "
                  f"hours over 35 C by cell {r['hours_over_35_min_pct']}-"
                  f"{r['hours_over_35_max_pct']}%", flush=True)
        (out / "HEAT_SUMMARY.json").write_text(json.dumps(
            {"box": BOX, "variable": HEAT_VAR, "units": "degrees C",
             "source": f"NOAA AORC v1.1 via s3://{BUCKET}, anonymous, public "
                       f"domain", "heatwaves": report}, indent=1))
        print(f"-> {out}")
        return 0

    want = args.storm or list(STORMS)
    report = []
    for name in want:
        if name not in STORMS:
            print(f"unknown storm {name!r}; have {list(STORMS)}")
            return 2
        start, end = STORMS[name]
        print(f"[aorc] {name} {start}..{end}", flush=True)
        r = fetch(name, start, end, out)
        report.append(r)
        print(f"       {r['grid']}  mean total {r['total_mm_mean']} mm, "
              f"wettest cell {r['total_mm_max']} mm, "
              f"peak hour {r['peak_hour_mm']} mm/h "
              f"(wettest cell-hour {r['peak_cell_mm_h']})", flush=True)
    (out / "SUMMARY.json").write_text(json.dumps(
        {"box": BOX, "variable": VAR, "units": "mm per hour",
         "source": f"NOAA AORC v1.1 via s3://{BUCKET}, anonymous, public domain",
         "storms": report}, indent=1))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
