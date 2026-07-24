"""Phase 5: the end-to-end loop, transfer gap, and the results table."""

from __future__ import annotations

import torch

from eval.metrics import markdown_table, to_rows
from pspe.envs import make_env
from pspe.perceive import PerceiveConfig, PerceiveModule
from pspe.pipeline import PSPEPipeline, transfer_gap
from pspe.plan import GaussianFieldPolicy
from pspe.simulate import make_surrogate

GRID = 16


def test_pipeline_rollout_reports_every_section7_metric() -> None:
    env = make_env("dar", grid=GRID, horizon=3, batched=True)
    policy = GaussianFieldPolicy(env.obs_shape[0], env.action_dim)
    perceive = PerceiveModule(PerceiveConfig(image_size=GRID, out_grid=GRID))
    pipeline = PSPEPipeline(env, policy, perceive=perceive)

    report = pipeline.rollout(batch=2, generator=torch.Generator().manual_seed(0))
    assert len(report.records) == 3

    summary = report.summary(env.task.cost_limit)
    for key in ("e2e/return", "e2e/episode_cost", "e2e/perception_rel_l2",
                "e2e/faithfulness", "e2e/violated"):
        assert key in summary
    assert 0.0 <= summary["e2e/faithfulness"] <= 1.0
    assert report.records[0].brief


def test_perception_off_is_the_perfect_perception_upper_bound() -> None:
    env = make_env("dar", grid=GRID, horizon=2, batched=True)
    policy = GaussianFieldPolicy(env.obs_shape[0], env.action_dim)
    pipeline = PSPEPipeline(env, policy, perceive=None, use_perception=False)
    report = pipeline.rollout(batch=2, generator=torch.Generator().manual_seed(0))
    assert all(r.perception_rel_l2 == 0.0 for r in report.records)


def test_cross_family_transfer_reports_incompatible_arity() -> None:
    surrogate = make_surrogate("fno", 1, grid=GRID, modes=4, width=8, n_layers=1)
    gap = transfer_gap(surrogate, "dar", "rdf", grid=GRID, steps=2, batch=2)
    # dar has 1 channel, rdf has 2: the helper must say so instead of guessing.
    assert "note" in gap


def test_parameter_shift_transfer_is_zero_against_the_source_regime() -> None:
    from pspe.simulate.solvers import DiffusionAdvectionReaction

    surrogate = make_surrogate("fno", 1, grid=GRID, modes=4, width=8, n_layers=1)
    identical = transfer_gap(
        surrogate, "dar", target_params=DiffusionAdvectionReaction.default_params(),
        grid=GRID, steps=2, batch=2,
    )
    assert abs(identical["transfer_gap"]) < 1e-6

    shifted = transfer_gap(surrogate, "dar", grid=GRID, steps=2, batch=2)
    assert shifted["protocol"].startswith("parameter shift")
    assert shifted["target_rel_l2"] != shifted["source_rel_l2"]


def test_results_table_renders() -> None:
    summaries = {
        "simulate/dar_fno": {"rel_l2_final": 0.0123, "wall_clock_s": 4.2},
        "plan/dar": {"return": -1.5, "violation_rate": 0.0},
    }
    table = markdown_table(to_rows(summaries))
    assert "rel L2 (rollout)" in table
    assert "violation rate" in table
    assert table.count("\n") >= 3


def test_padded_surrogate_accepts_every_testbed_arity() -> None:
    """One head, three channel counts — the precondition for cross-family transfer."""
    from pspe.simulate import make_padded_surrogate, supports_cross_family
    from pspe.simulate.solvers import make_testbed

    model = make_padded_surrogate("fno", grid=GRID, modes=4, width=8, n_layers=1)
    assert supports_cross_family(model)

    for name in ("dar", "rdf", "swe"):
        testbed = make_testbed(name, grid=GRID)
        state = testbed.initial_condition(2, torch.Generator().manual_seed(0))
        out = model(state)
        assert out.shape == state.shape, f"{name}: {out.shape} != {state.shape}"


def test_cross_family_transfer_is_scored_with_a_padded_surrogate() -> None:
    from pspe.simulate import make_padded_surrogate

    model = make_padded_surrogate("fno", grid=GRID, modes=4, width=8, n_layers=1)
    gap = transfer_gap(model, "dar", "swe", grid=GRID, steps=2, batch=2)

    # The arity excuse no longer applies: this must be a real number.
    assert "note" not in gap
    assert gap["protocol"] == "cross-family dar->swe"
    assert isinstance(gap["transfer_gap"], float)
    assert gap["transfer_gap"] == gap["target_rel_l2"] - gap["source_rel_l2"]
