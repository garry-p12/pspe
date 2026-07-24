"""Phase 1: solvers, surrogates, losses and the fidelity metric."""

from __future__ import annotations

import pytest
import torch

from pspe.envs.actuators import GaussianActuatorBasis
from pspe.simulate import (
    PDEDataset,
    make_surrogate,
    make_testbed,
    relative_l2,
    rollout_truth,
    surrogate_fidelity,
    surrogate_loss,
)
from pspe.simulate.dataset import DataConfig, generate_dataset
from pspe.simulate.solvers import TESTBEDS, laplacian

GRID = 32


@pytest.mark.parametrize("name", sorted(TESTBEDS))
def test_solver_is_stable_over_a_full_episode(name: str) -> None:
    testbed = make_testbed(name, grid=GRID)
    generator = torch.Generator().manual_seed(0)
    state = testbed.initial_condition(2, generator)
    basis = GaussianActuatorBasis(9, GRID)
    control = basis.field(basis.sample_amplitudes(2, 0.5, generator))

    for _ in range(24):
        state = testbed.step(state, control)
    assert torch.isfinite(state).all()
    assert state.abs().max() < 1e3, "solver blew up"


@pytest.mark.parametrize("name", sorted(TESTBEDS))
def test_true_transition_has_small_residual(name: str) -> None:
    """The residual of a solver step should be discretisation-level, not O(1)."""
    testbed = make_testbed(name, grid=GRID)
    state = testbed.initial_condition(2, torch.Generator().manual_seed(0))
    next_state = testbed.step(state)
    residual = (testbed.residual(state, next_state) ** 2).mean().sqrt()
    delta = ((next_state - state) ** 2).mean().sqrt() / testbed.dt
    assert float(residual) < 0.2 * float(delta)


def test_laplacian_matches_analytic_solution() -> None:
    """lap(sin 2pi x) = -(2pi)^2 sin 2pi x on a periodic grid."""
    grid = 128
    lin = torch.linspace(0, 1, grid + 1)[:-1]
    xx = lin.view(1, 1, 1, -1).expand(1, 1, grid, grid)
    u = torch.sin(2 * torch.pi * xx)
    got = laplacian(u, 1.0 / grid)
    want = -((2 * torch.pi) ** 2) * u
    assert float(relative_l2(got, want)) < 0.01


@pytest.mark.parametrize("kind", ["fno", "deeponet", "gnot"])
def test_surrogate_shapes_and_gradients(kind: str) -> None:
    testbed = make_testbed("dar", grid=GRID)
    model = make_surrogate(kind, testbed.n_channels, grid=GRID)
    state = testbed.initial_condition(2, torch.Generator().manual_seed(0))
    control = torch.zeros(2, 1, GRID, GRID, requires_grad=True)

    out = model(state, control)
    assert out.shape == state.shape
    out.sum().backward()
    # The planner's pathwise branch needs d(next state)/d(control) to exist.
    assert control.grad is not None and torch.isfinite(control.grad).all()


def test_surrogate_loss_decreases_on_a_tiny_fit() -> None:
    testbed = make_testbed("dar", grid=GRID)
    model = make_surrogate("fno", 1, grid=GRID, modes=6, width=16, n_layers=2)
    generator = torch.Generator().manual_seed(0)
    u0 = testbed.initial_condition(4, generator)
    controls = torch.zeros(4, 3, 1, GRID, GRID)
    traj = rollout_truth(testbed, u0, controls, 3)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = last = None
    for step in range(15):
        loss, components = surrogate_loss(model, testbed, traj, controls, horizon=2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 0:
            first = components["loss/total"]
        last = components["loss/total"]
    assert last < first


def test_physics_term_is_zero_weighted_when_disabled() -> None:
    testbed = make_testbed("dar", grid=GRID)
    model = make_surrogate("fno", 1, grid=GRID, modes=4, width=8, n_layers=1)
    u0 = testbed.initial_condition(2, torch.Generator().manual_seed(0))
    controls = torch.zeros(2, 2, 1, GRID, GRID)
    traj = rollout_truth(testbed, u0, controls, 2)

    _, off = surrogate_loss(model, testbed, traj, controls, 2, use_physics=False)
    _, on = surrogate_loss(model, testbed, traj, controls, 2, use_physics=True)
    assert off["loss/physics"] == 0.0
    assert on["loss/physics"] > 0.0


def test_fidelity_report_is_monotone_in_horizon() -> None:
    testbed = make_testbed("dar", grid=GRID)
    model = make_surrogate("fno", 1, grid=GRID, modes=4, width=8, n_layers=1)
    u0 = testbed.initial_condition(2, torch.Generator().manual_seed(0))
    controls = torch.zeros(2, 4, 1, GRID, GRID)
    report = surrogate_fidelity(model, testbed, u0, controls, 4)
    assert report["rel_l2_final"] >= report["rel_l2_1step"] - 1e-6
    assert len(report["rel_l2_per_step"]) == 4


def test_dataset_roundtrip(tmp_path) -> None:
    cfg = DataConfig(testbed="dar", n_trajectories=8, steps=6, grid=GRID, control_hold=2)
    data = generate_dataset(cfg, out_path=tmp_path / "dar.npz")
    assert data["states"].shape == (8, 7, 1, GRID, GRID)
    assert data["controls"].shape == (8, 6, 1, GRID, GRID)

    train = PDEDataset(data["states"], data["controls"], horizon=2, split="train")
    window, control = train[0]
    assert window.shape == (3, 1, GRID, GRID)
    assert control.shape == (2, 1, GRID, GRID)

    # The control channel must actually be excited, or the surrogate can never
    # learn the counterfactual the planner asks it for.
    assert float(abs(data["controls"]).mean()) > 1e-3
