"""Surveyed floor, gate and road levels for 2,600+ Lismore properties.

Everything the portal has said about depth so far is depth above the GROUND,
because that is all a DEM can give. What a resident, an insurer and a council
actually ask is depth above the FLOOR, and the gap between those two is not a
constant: measured here, the median Lismore floor stands 2.0 m above its own
front gate, with the middle 90% spanning 0.04 m to 3.3 m. A slab-on-ground shop
and a Queenslander on stumps in the same street flood at stages three metres
apart, and no elevation model can tell them apart.

Lismore City Council surveyed those floors. The levels were first recorded in
December 1996, updated in March 2006 and at various times since (the most
recent column in the source documents is August 2018), and published per
address as four suburb PDFs. Each row carries, in metres AHD:

    Floor Level    the habitable floor
    Gate Level     ground at the front boundary
    Road Centre    the road crown outside
    1in10/1in100   the council's design flood levels at that property

The last two make this more than an input. `Gate Level` and `Road Centre` are
surveyed ground elevations at thousands of points inside a box the solver
already runs on, which is a check on the 30 m Copernicus DEM that does not come
from another model. The design levels are a check on the solved water surface
that does not come from us either.

**Datum.** These are AHD. The Copernicus DEM is EGM2008. Over northern NSW the
two differ by a few tenths of a metre, and the difference is not zero, so
anything that subtracts one from the other has to say so -- see
`docs/THESIS_AND_RESULTS.md` §5.14e. This script does no conversion; it
records the datum and leaves the correction to whatever uses the numbers.

**Provenance.** The council's own disclaimer travels with the data and is
reproduced in the output directory: most of the levels were supplied to Council
rather than surveyed by it, and Council does not endorse them as free of error.
Treat them as the best available record of floor heights, not as a survey.

Usage:
    python3 scripts/fetch_lismore_floor_levels.py --out data/lismore
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.request
from pathlib import Path

# The council's documents, as published. bigrob.com.au serves the same files
# but its certificate has expired, so the mirror is the one used.
BASE = "https://bigrob.summerlandlegal.com.au/assets/files"
SUBURB_FILES = {
    "Lismore": "Floor-andFloodLevels-Lismore.pdf",
    "North Lismore": "Floor-andFloodLevels-NorthLismore.pdf",
    "East Lismore & Girards Hill": "Floor-andFloodLevels-EastLismoreGirardsHill.pdf",
    "South Lismore": "Floor-andFloodLevels-SouthLismore.pdf",
}

# Longest first: "EAST LISMORE" has to win over "LISMORE", or every East
# Lismore address is filed under the wrong suburb.
SUBURBS = sorted(
    ["EAST LISMORE", "NORTH LISMORE", "SOUTH LISMORE", "GIRARDS HILL",
     "LISMORE HEIGHTS", "LISMORE", "GOONELLABAH", "MONALTRIE", "LOFTVILLE"],
    key=len, reverse=True,
)

# Three decimals occur: one East Lismore row carries 13.187. Allowing only one
# or two shifted that row's columns by one and put a level in the suburb.
NUM = r"\d{1,3}\.\d{1,3}"
ROW = re.compile(
    rf"^\s*(?P<addr>\S.*?)\s+(?P<floor>{NUM})\s+(?P<gate>{NUM})\s+(?P<road>{NUM})\s*(?P<rest>.*)$")
LEADING_DATE = re.compile(
    r"^(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+)?\d{4}\s+")
SUBURB_RE = re.compile(rf"\s+({'|'.join(SUBURBS)})\s*$", re.IGNORECASE)


def parse_pdf(path: Path) -> list[dict]:
    import pypdf                                    # noqa: PLC0415

    out: list[dict] = []
    reader = pypdf.PdfReader(str(path))
    for pno, page in enumerate(reader.pages, 1):
        for line in (page.extract_text() or "").split("\n"):
            m = ROW.match(line)
            if not m:
                continue
            addr = LEADING_DATE.sub("", m.group("addr").strip())
            s = SUBURB_RE.search(addr)
            if not s:
                continue                            # a header or a stray line
            street = addr[:s.start()].strip()
            mnum = re.match(r"^([\w/\-]+)\s+(.+)$", street)
            if not mnum:
                continue
            tail = [float(x) for x in re.findall(NUM, m.group("rest"))]
            out.append({
                "number": mnum.group(1),
                "street": mnum.group(2).strip(),
                "suburb": s.group(1).upper(),
                "floor_m_ahd": float(m.group("floor")),
                "gate_m_ahd": float(m.group("gate")),
                "road_m_ahd": float(m.group("road")),
                "design_1in10_m_ahd": tail[0] if len(tail) >= 2 else "",
                "design_1in100_m_ahd": tail[-1] if tail else "",
                "description": re.sub(rf"\s*{NUM}\s*", " ", m.group("rest")).strip(),
                "source_pdf": path.name,
                "source_page": pno,
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/lismore")
    ap.add_argument("--keep-pdfs", action="store_true",
                    help="leave the source PDFs in place for re-parsing")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "pdf").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for suburb, fname in SUBURB_FILES.items():
        pdf = out / "pdf" / fname
        if not pdf.exists():
            url = f"{BASE}/{fname}"
            print(f"fetching {url}")
            req = urllib.request.Request(
                url, headers={"User-Agent": "PSPE-floodplain-planner/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                pdf.write_bytes(r.read())
        got = parse_pdf(pdf)
        print(f"  {suburb:30s} {len(got):5d} properties")
        rows += got

    csv_path = out / "floor_levels.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    (out / "PROVENANCE.json").write_text(json.dumps({
        "dataset": "Lismore flood and floor levels",
        "publisher": "Lismore City Council",
        "mirror": BASE,
        "files": list(SUBURB_FILES.values()),
        "properties": len(rows),
        "datum": "AHD (Australian Height Datum)",
        "surveyed": "first recorded December 1996; updated March 2006 and at "
                    "various times since, latest column August 2018",
        "disclaimer": "Council states the information was provided from various "
                      "sources, that only some was verified by Council survey, "
                      "and that Council does not endorse it as free of error, "
                      "omission or misdescription.",
        "caution": "AHD is not the Copernicus DEM's EGM2008. Do not subtract "
                   "one from the other without stating the offset.",
    }, indent=2) + "\n")

    if not args.keep_pdfs:
        for p in (out / "pdf").glob("*.pdf"):
            p.unlink()
        (out / "pdf").rmdir()

    print(f"\n{len(rows)} properties -> {csv_path}")


if __name__ == "__main__":
    main()
