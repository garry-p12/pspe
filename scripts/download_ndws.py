#!/usr/bin/env python
"""Fetch the Next Day Wildfire Spread dataset from Kaggle.

    python scripts/download_ndws.py                 # all splits (~1-2 GB)
    python scripts/download_ndws.py --split eval    # just the eval split

Needs a Kaggle API token: create one at kaggle.com -> Settings -> API ->
"Create New Token", then save the downloaded kaggle.json at ~/.kaggle/kaggle.json
with mode 600. The dataset itself is public; the token only identifies you to
the API.

No credential is read, stored, or transmitted by this repository beyond handing
the standard Kaggle client its own config file.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pspe.simulate.real.ndws import DATA_ROOT, SPLIT_PREFIX  # noqa: E402

DATASET = "fantineh/next-day-wildfire-spread"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default=None, choices=sorted(SPLIT_PREFIX),
                        help="download only this split (default: everything)")
    parser.add_argument("--out", default=str(DATA_ROOT))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    token = Path.home() / ".kaggle" / "kaggle.json"
    if not token.exists() and not os.environ.get("KAGGLE_USERNAME"):
        print(
            "No Kaggle credentials found.\n"
            f"  expected {token}, or KAGGLE_USERNAME/KAGGLE_KEY in the environment\n"
            "  get a token at kaggle.com -> Settings -> API -> Create New Token",
            file=sys.stderr,
        )
        return 2

    cmd = ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(out), "--unzip"]
    if args.split:
        # -f takes one file; the split spans several shards, so filter after the
        # fact rather than guessing shard counts here.
        print(f"note: downloading the full archive, then keeping only '{args.split}'")

    print(" ".join(cmd))
    try:
        proc = subprocess.run(cmd)
    except FileNotFoundError:
        print("`kaggle` CLI not installed: pip install kaggle", file=sys.stderr)
        return 2
    if proc.returncode != 0:
        return proc.returncode

    if args.split:
        keep = SPLIT_PREFIX[args.split]
        for path in out.glob("*.tfrecord"):
            if not path.name.startswith(keep):
                path.unlink()

    files = sorted(out.glob("*.tfrecord"))
    total = sum(f.stat().st_size for f in files) / 1e6
    print(f"{len(files)} tfrecord file(s), {total:.0f} MB in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
