"""Real wildfire data: TFRecord parsing and the NDWS -> schema mapping.

Fixtures are genuine TFRecords — correct framing, correct CRC32C — written by
this repository's own writer, so the reader is tested against the format rather
than against itself. `h5py`-style `importorskip` is deliberately absent: these
tests must fail loudly if the parser breaks, not skip.
"""

from __future__ import annotations

import numpy as np
import pytest

from pspe.simulate.real import fire_stats, load_ndws
from pspe.simulate.real.ndws import DRIVER_FEATURES, NDWSConfig
from pspe.simulate.real.tfrecord import (
    crc32c,
    iter_examples,
    mask_crc,
    parse_example,
    read_records,
    write_example,
    write_records,
)

GRID = 8


def make_example(prev: np.ndarray, nxt: np.ndarray, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    features = {"PrevFireMask": prev, "FireMask": nxt}
    for name in DRIVER_FEATURES:
        features[name] = rng.normal(size=(GRID, GRID)).astype(np.float32)
    return write_example(features)


def write_fixture(path, n: int = 3, unlabelled: bool = False):
    payloads = []
    for i in range(n):
        prev = np.zeros((GRID, GRID), np.float32)
        prev[2:4, 2:4] = 1.0
        nxt = np.zeros((GRID, GRID), np.float32)
        nxt[2:5, 2:5] = 1.0                      # the fire grew
        if unlabelled:
            nxt[0, :] = -1.0                     # a row of missing labels
        payloads.append(make_example(prev, nxt, seed=i))
    write_records(path, payloads)
    return path


# --------------------------------------------------------------------------- #
# Format
# --------------------------------------------------------------------------- #
def test_crc32c_matches_known_vector() -> None:
    """Castagnoli, not the usual CRC32 — a wrong table would still self-agree."""
    assert crc32c(b"123456789") == 0xE3069283
    assert mask_crc(crc32c(b"")) != crc32c(b"")


def test_roundtrip_through_genuine_framing(tmp_path) -> None:
    path = write_fixture(tmp_path / "a.tfrecord", n=2)
    payloads = list(read_records(path, verify=True))   # CRCs checked
    assert len(payloads) == 2

    example = parse_example(payloads[0])
    assert "FireMask" in example and "PrevFireMask" in example
    assert example["FireMask"].size == GRID * GRID


def test_corrupt_payload_is_detected(tmp_path) -> None:
    path = write_fixture(tmp_path / "b.tfrecord", n=1)
    raw = bytearray(path.read_bytes())
    raw[-8] ^= 0xFF                                   # flip a byte inside the payload
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="CRC mismatch"):
        list(read_records(path, verify=True))


def test_float_values_survive_the_roundtrip(tmp_path) -> None:
    values = np.linspace(-3, 3, GRID * GRID, dtype=np.float32).reshape(GRID, GRID)
    path = tmp_path / "c.tfrecord"
    write_records(path, [write_example({"FireMask": values, "PrevFireMask": values})])
    parsed = next(iter_examples(path))
    np.testing.assert_allclose(parsed["FireMask"].reshape(GRID, GRID), values, rtol=1e-6)


# --------------------------------------------------------------------------- #
# Schema mapping
# --------------------------------------------------------------------------- #
def test_load_produces_the_repo_schema(tmp_path) -> None:
    write_fixture(tmp_path / "next_day_wildfire_spread_train_00.tfrecord", n=3)
    data = load_ndws(NDWSConfig(root=tmp_path, grid=GRID))

    assert data["states"].shape == (3, 2, 1, GRID, GRID)
    assert data["controls"].shape == data["states"].shape
    assert data["drivers"].shape == (3, len(DRIVER_FEATURES), GRID, GRID)
    # No intervention is recorded in observational data, and pretending
    # otherwise is how a planning claim gets made on data that cannot support one.
    assert np.all(data["controls"] == 0.0)


def test_unlabelled_cells_are_masked_not_zero_filled(tmp_path) -> None:
    """-1 means "unknown", and must never be read as "no fire"."""
    write_fixture(tmp_path / "train.tfrecord", n=2, unlabelled=True)
    data = load_ndws(NDWSConfig(root=tmp_path, grid=GRID))

    next_valid = data["valid"][:, 1, 0]
    assert not next_valid[:, 0, :].any(), "unlabelled row should be invalid"
    assert next_valid[:, 1:, :].all(), "labelled rows should be valid"
    # The raw -1 survives in `states`: a caller ignoring `valid` gets an
    # obviously wrong number rather than a plausible one.
    assert data["states"][:, 1, 0][:, 0, :].min() == -1.0


def test_fire_stats_reports_prevalence_and_coverage(tmp_path) -> None:
    write_fixture(tmp_path / "train.tfrecord", n=4, unlabelled=True)
    stats = fire_stats(load_ndws(NDWSConfig(root=tmp_path, grid=GRID)))

    assert stats["samples"] == 4
    assert 0.0 < stats["labelled_fraction"] < 1.0
    # Rare positives: without this number, any headline score is unreadable.
    assert 0.0 < stats["fire_prevalence"] < 0.5


def test_n_samples_caps_the_load(tmp_path) -> None:
    write_fixture(tmp_path / "train.tfrecord", n=5)
    data = load_ndws(NDWSConfig(root=tmp_path, grid=GRID, n_samples=2))
    assert data["states"].shape[0] == 2


def test_missing_data_says_how_to_get_it(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="download_ndws"):
        load_ndws(NDWSConfig(root=tmp_path / "empty", grid=GRID))


def test_driver_normalisation_puts_channels_on_one_scale(tmp_path) -> None:
    """Raw channels differ by orders of magnitude; unnormalised, the big ones win."""
    from pspe.simulate.real import driver_stats, normalize_drivers

    write_fixture(tmp_path / "train.tfrecord", n=6)
    data = load_ndws(NDWSConfig(root=tmp_path, grid=GRID))
    # Blow one channel up to elevation-like magnitudes.
    data["drivers"][:, 0] *= 5000.0

    stats = driver_stats(data)
    normed = normalize_drivers(data["drivers"], stats)

    assert np.abs(normed).max() <= float(stats["clip"]) + 1e-6
    spread = normed.reshape(normed.shape[1], -1).std(axis=1) if normed.shape[0] == 1 \
        else normed.transpose(1, 0, 2, 3).reshape(normed.shape[1], -1).std(axis=1)
    assert spread.max() / max(spread.min(), 1e-6) < 50, "channels still differ wildly in scale"


def test_normalisation_stats_come_from_one_split_only(tmp_path) -> None:
    """Fitting the transform on eval too would leak eval into training."""
    from pspe.simulate.real import driver_stats, normalize_drivers

    write_fixture(tmp_path / "train.tfrecord", n=4)
    data = load_ndws(NDWSConfig(root=tmp_path, grid=GRID))
    stats = driver_stats(data)

    shifted = data["drivers"] + 100.0          # a "different split"
    applied = normalize_drivers(shifted, stats)
    # The transform must be the train one, so a shifted split does NOT recentre.
    assert applied.mean() > 1.0, "stats were refit instead of reused"
