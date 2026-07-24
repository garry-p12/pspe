#!/usr/bin/env python
"""Download a PDEBench 2D file into data/pdebench/.

    python scripts/download_pdebench.py 2d_reacdiff
    python scripts/download_pdebench.py 2d_shallowwater

PDEBench is hosted on DaRUS (doi:10.18419/darus-2986), open access, no account.
Files are large (multi-GB); this streams to disk with a progress line. On a
constrained box the download may be slow — the loader and its tests do not need
it, and the Colab notebook can run this cell instead.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pspe.simulate.pdebench import DATA_ROOT, PDEBENCH_FILES  # noqa: E402


def _hook(count: int, block: int, total: int) -> None:
    if total > 0:
        pct = min(100, count * block * 100 // total)
        mb = count * block / 1e6
        sys.stdout.write(f"\r  {pct:3d}%  {mb:8.1f} MB")
        sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=sorted(PDEBENCH_FILES))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    spec = PDEBENCH_FILES[args.name]
    out = Path(args.out) if args.out else DATA_ROOT / spec["file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"already present: {out}")
        return 0

    print(f"{args.name}: {spec['file']}  ({spec['size']}, maps to testbed "
          f"'{spec['maps_to']}', {spec['channels']})")
    print(f"  from {spec['url']}")
    if args.out is None:
        print("  NOTE: multi-GB download — best run on Colab or a box with open, "
              "fast network.")
    try:
        urllib.request.urlretrieve(spec["url"], out, _hook)
        print(f"\n  -> {out}")
    except Exception as exc:  # network / host issues are common on constrained boxes
        print(f"\n  download failed: {exc}")
        print("  Run this on Colab or a box with open network; the loader and its "
              "tests do not require the file.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
