"""Loaders for real observational datasets.

Each maps an external source into the `(states, controls)` schema the synthetic
testbeds use, so the training code downstream does not know the difference.
"""

from .ndws import (
    NDWSConfig,
    driver_stats,
    fire_stats,
    load_ndws,
    normalize_drivers,
)
from .tfrecord import iter_examples, parse_example, read_records

__all__ = [
    "NDWSConfig",
    "load_ndws",
    "fire_stats",
    "driver_stats",
    "normalize_drivers",
    "iter_examples",
    "parse_example",
    "read_records",
]
