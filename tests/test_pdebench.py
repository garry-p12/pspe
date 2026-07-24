"""PDEBench loader: layout normalisation + a real HDF5 round-trip.

The download runs on Colab, but the correctness-bearing logic — turning
PDEBench's several array layouts into the repo's (states, controls) schema, and
reading a genuine HDF5 file — is tested here. `test_hdf5_roundtrip` writes an
actual PDEBench-shaped .h5 with h5py and loads it back, so the real code path
(not a mock) is exercised offline.
"""

from __future__ import annotations

import numpy as np
import pytest

from pspe.simulate.pdebench import PDEBenchConfig, load_pdebench, normalise_layout, to_schema


def test_normalise_layout_handles_every_pdebench_shape() -> None:
    # (n, T, H, W) -> add channel
    assert normalise_layout(np.zeros((5, 10, 32, 32))).shape == (5, 10, 1, 32, 32)
    # (n, T, H, W, C) channel-last -> channel moved in
    assert normalise_layout(np.zeros((5, 10, 32, 32, 2))).shape == (5, 10, 2, 32, 32)
    # (T, H, W) single trajectory -> add sample + channel
    assert normalise_layout(np.zeros((10, 32, 32))).shape == (1, 10, 1, 32, 32)
    # already (n, T, C, H, W) with a non-square-triggering layout is passed through
    already = np.zeros((3, 8, 2, 16, 16))
    assert normalise_layout(already).shape == (3, 8, 2, 16, 16)


def test_normalise_layout_rejects_unhandled_rank() -> None:
    with pytest.raises(ValueError):
        normalise_layout(np.zeros((4, 4)))


def test_to_schema_downsamples_and_zeros_controls() -> None:
    # A (n, T, H, W, C) PDEBench block at 96px, downsampled to 32.
    arr = np.random.rand(6, 20, 96, 96, 2).astype(np.float32)
    out = to_schema(arr, grid=32, n_samples=4, max_steps=12)

    assert out["states"].shape == (4, 13, 2, 32, 32)     # n capped, T = max_steps+1
    assert out["controls"].shape == (4, 12, 1, 32, 32)   # steps = T-1, single channel
    # PDEBench has no forcing: controls must be exactly zero.
    assert np.all(out["controls"] == 0.0)


def test_hdf5_roundtrip(tmp_path) -> None:
    """Write a real PDEBench-layout HDF5 and load it through the actual reader."""
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "2D_diff-react_NA_NA.h5"

    truth = np.random.rand(8, 16, 48, 48, 2).astype(np.float32)  # (n,T,H,W,C)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=truth)
        # PDEBench files also carry a grid group; the loader must ignore it.
        grid = f.create_group("grid")
        grid.create_dataset("x", data=np.linspace(0, 1, 48))
        grid.create_dataset("t", data=np.linspace(0, 1, 16))

    out = load_pdebench(PDEBenchConfig(path=str(path), grid=32, n_samples=8, max_steps=10))
    assert out["states"].shape == (8, 11, 2, 32, 32)
    assert out["controls"].shape == (8, 10, 1, 32, 32)
    assert np.isfinite(out["states"]).all()


def test_hdf5_per_sample_groups(tmp_path) -> None:
    """The other PDEBench layout: one group per sample, each with a 'data' array."""
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "per_sample.h5"
    with h5py.File(path, "w") as f:
        for i in range(4):
            g = f.create_group(f"{i:04d}")
            g.create_dataset("data", data=np.random.rand(12, 24, 24, 1).astype(np.float32))

    out = load_pdebench(PDEBenchConfig(path=str(path), grid=16, n_samples=4, max_steps=8))
    assert out["states"].shape == (4, 9, 1, 16, 16)


def test_missing_file_gives_a_helpful_error() -> None:
    with pytest.raises(FileNotFoundError, match="download_pdebench"):
        load_pdebench(PDEBenchConfig(path="/nonexistent/pdebench.h5"))


def test_data_based_eval_scores_against_stored_frames(tmp_path) -> None:
    """PDEBench has no solver, so eval_vs_data must roll vs recorded frames."""
    import numpy as np
    import torch

    from pspe.simulate import SimulateTrainConfig, SimulateTrainer, make_surrogate
    from pspe.utils import RunLogger

    # Smooth spreading blobs a tiny FNO can fit; no analytic testbed involved.
    n, t, h = 24, 10, 16
    g = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(g, g)
    states = np.zeros((n, t, 1, h, h), dtype=np.float32)
    rng = np.random.default_rng(0)
    for i in range(n):
        cx, cy = rng.uniform(0.3, 0.7, 2)
        for k in range(t):
            s = 0.05 + 0.02 * k
            states[i, k, 0] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
    blob = {"states": states, "controls": np.zeros((n, t - 1, 1, h, h), dtype=np.float32)}

    model = make_surrogate("fno", 1, grid=h, modes=6, width=16, n_layers=2)  # width % 8 == 0
    trainer = SimulateTrainer(
        model,
        SimulateTrainConfig(testbed="swe", surrogate="fno", epochs=6, grid=h,
                            eval_vs_data=True),
        blob, RunLogger(tmp_path / "pdb", use_tensorboard=False),
    )
    summary = trainer.train()
    # A solver-based eval would have crashed on the channel mismatch; data eval
    # just compares rolled predictions to stored frames. Loose bound: this is a
    # 16px, 6-epoch smoke fit — the point is the data-eval path runs and beats a
    # trivial predictor, not that it converges.
    assert summary["rel_l2_1step"] < 0.5
    assert "rel_l2_final" in summary
