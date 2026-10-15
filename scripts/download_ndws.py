#!/usr/bin/env python
"""Fetch the Next Day Wildfire Spread dataset from Kaggle.

    python scripts/download_ndws.py                 # all splits (~1-2 GB)
    python scripts/download_ndws.py --split eval    # just the eval split

Needs a Kaggle API token: create one at kaggle.com -> Settings -> API ->
"Create New Token". Recent tokens arrive as ~/.kaggle/access_token, which the
`kaggle` CLI does not read; `kagglehub` does, and is tried first. Older
kaggle.json and KAGGLE_USERNAME/KAGGLE_KEY both still work.

The dataset itself is public; the token only identifies you to the API.

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

    # Try kagglehub first. It accepts the credential forms the `kaggle` CLI
    # does AND ~/.kaggle/access_token, which is what a recent "Create New
    # Token" actually writes -- the CLI-only path below reported "no Kaggle
    # credentials found" on a machine that was perfectly well authenticated,
    # and the dataset sat un-downloadable behind a wrong error message.
    #
    # It also caches by dataset version, so a second call after an accidental
    # deletion costs a symlink rather than 2 GB.
    try:
        import kagglehub                      # noqa: PLC0415
    except ImportError:
        kagglehub = None
    if kagglehub is not None:
        try:
            cached = Path(kagglehub.dataset_download(DATASET))
            shards = sorted(cached.rglob("*.tfrecord"))
            if shards:
                for src in shards:
                    if args.split and not src.name.startswith(SPLIT_PREFIX[args.split]):
                        continue
                    dst = out / src.name
                    if dst.is_symlink() or dst.exists():
                        dst.unlink()
                    # Symlink rather than copy: the cache already holds 2 GB
                    # and a second copy helps nobody.
                    dst.symlink_to(src)
                files = sorted(out.glob("*.tfrecord"))
                total = sum(f.stat().st_size for f in files) / 1e6
                print(f"{len(files)} tfrecord file(s), {total:.0f} MB linked into {out}")
                print(f"  cache: {cached}")
                return 0
            print("kagglehub returned no tfrecords; falling back to the CLI",
                  file=sys.stderr)
        except Exception as exc:                                # noqa: BLE001
            print(f"kagglehub could not fetch it ({str(exc)[:120]}); "
                  f"falling back to the CLI", file=sys.stderr)

    token = Path.home() / ".kaggle" / "kaggle.json"
    access = Path.home() / ".kaggle" / "access_token"
    if not token.exists() and not access.exists() and not os.environ.get("KAGGLE_USERNAME"):
        print(
            "No Kaggle credentials found.\n"
            f"  expected {token}, {access}, or KAGGLE_USERNAME/KAGGLE_KEY\n"
            "  get a token at kaggle.com -> Settings -> API -> Create New Token\n"
            "  and install one of: pip install kagglehub   (handles access_token)\n"
            "                      pip install kaggle      (needs kaggle.json)",
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
        print("Neither kagglehub nor the kaggle CLI is installed.\n"
              "  pip install kagglehub   (works with ~/.kaggle/access_token)",
              file=sys.stderr)
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
