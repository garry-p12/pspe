"""Phase F: seed aggregation — the arithmetic behind every error bar."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

from eval.metrics import aggregate_seeds, markdown_table

ROOT = Path(__file__).resolve().parents[1]


def test_mean_and_sample_std_match_statistics() -> None:
    per_seed = {
        0: [{"run": "plan alpha=adaptive", "return": -2.0, "episode_cost": 0.20}],
        1: [{"run": "plan alpha=adaptive", "return": -2.4, "episode_cost": 0.24}],
        2: [{"run": "plan alpha=adaptive", "return": -2.2, "episode_cost": 0.22}],
    }
    rows, stats = aggregate_seeds(per_seed)

    assert len(rows) == 1
    assert rows[0]["seeds"] == 3
    returns = [-2.0, -2.4, -2.2]
    assert stats["plan alpha=adaptive"]["return"]["mean"] == statistics.fmean(returns)
    # Sample std (ddof=1), not the population std: these are drawn runs.
    assert stats["plan alpha=adaptive"]["return"]["std"] == statistics.stdev(returns)
    assert "±" in rows[0]["return"]


def test_single_seed_reports_no_spread() -> None:
    rows, stats = aggregate_seeds({0: [{"run": "solo", "return": -1.0}]})
    assert stats["solo"]["return"]["std"] is None
    assert "±" not in rows[0]["return"], "one seed must not claim a +/- 0 spread"
    assert rows[0]["seeds"] == 1


def test_a_run_missing_from_one_seed_is_averaged_over_the_seeds_that_have_it() -> None:
    per_seed = {
        0: [{"run": "a", "return": 1.0}, {"run": "b", "return": 3.0}],
        1: [{"run": "a", "return": 2.0}],
    }
    rows, stats = aggregate_seeds(per_seed)
    by_run = {row["run"]: row for row in rows}

    assert by_run["a"]["seeds"] == 2
    assert by_run["b"]["seeds"] == 1, "b ran once; the table must say so"
    assert stats["b"]["return"]["mean"] == 3.0
    assert stats["a"]["return"]["mean"] == 1.5


def test_metric_present_in_only_some_seeds_keeps_its_own_n() -> None:
    per_seed = {
        0: [{"run": "a", "return": 1.0, "final_alpha": 0.5}],
        1: [{"run": "a", "return": 2.0}],
    }
    _, stats = aggregate_seeds(per_seed)
    assert stats["a"]["return"]["n"] == 2
    assert stats["a"]["final_alpha"]["n"] == 1


def test_non_numeric_cells_collapse_only_when_they_agree() -> None:
    per_seed = {
        0: [{"run": "t", "protocol": "cross-family dar->swe", "surrogate": "fno"}],
        1: [{"run": "t", "protocol": "cross-family dar->swe", "surrogate": "deeponet"}],
    }
    rows, _ = aggregate_seeds(per_seed)
    assert rows[0]["protocol"] == "cross-family dar->swe"
    assert rows[0]["surrogate"] == "mixed"


def test_bools_average_to_a_fraction() -> None:
    per_seed = {
        0: [{"run": "explain", "backbone_is_stub": True}],
        1: [{"run": "explain", "backbone_is_stub": False}],
    }
    _, stats = aggregate_seeds(per_seed)
    assert stats["explain"]["backbone_is_stub"]["mean"] == 0.5


def test_aggregated_rows_render_as_a_table() -> None:
    rows, _ = aggregate_seeds({
        0: [{"run": "a", "return": 1.0}],
        1: [{"run": "a", "return": 2.0}],
    })
    table = markdown_table(rows)
    assert table.startswith("| run | seeds | return |")


def test_aggregate_only_tables_existing_seed_dirs(tmp_path: Path) -> None:
    for seed, value in ((0, 1.0), (1, 3.0)):
        out = tmp_path / f"seed_{seed}"
        out.mkdir()
        (out / "results.json").write_text(json.dumps([{"run": "plan", "return": value}]))

    proc = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "run_seeds.py"),
         "--aggregate-only", "--seeds", "0", "1", "--out", str(tmp_path)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr

    stats = json.loads((tmp_path / "results_seeds.json").read_text())["stats"]
    assert stats["plan"]["return"]["mean"] == 2.0
    assert stats["plan"]["return"]["std"] == statistics.stdev([1.0, 3.0])
    assert "±" in (tmp_path / "results_seeds.md").read_text()


def test_duplicate_seeds_are_refused(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "run_seeds.py"),
         "--seeds", "0", "0", "--out", str(tmp_path)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode != 0
    assert "duplicate seeds" in proc.stderr


def test_constant_columns_do_not_print_a_zero_spread() -> None:
    per_seed = {
        0: [{"run": "a", "return": 1.0, "params/total": 1186465}],
        1: [{"run": "a", "return": 2.0, "params/total": 1186465}],
    }
    rows, stats = aggregate_seeds(per_seed)
    assert rows[0]["params/total"] == "1.186e+06", "an identical column is not a +/- 0 measurement"
    assert stats["a"]["params/total"]["std"] == 0.0, "the raw std stays in the JSON"
    assert "±" in rows[0]["return"]


def test_transfer_matrix_layout_aggregates_through_the_same_path(tmp_path: Path) -> None:
    """`run_transfer.py` writes {matrix, in_family, pairs}; only `pairs` is rows."""
    for seed, gap in ((0, 1.5), (1, 2.5)):
        out = tmp_path / f"seed_{seed}"
        out.mkdir()
        (out / "transfer_matrix.json").write_text(json.dumps({
            "matrix": {"dar": {"swe": 38.1}},
            "in_family": {"dar": 0.05},
            "pairs": [{"run": "dar -> swe", "transfer gap": gap,
                       "protocol": "cross-family dar->swe"}],
        }))

    proc = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "run_seeds.py"),
         "--aggregate-only", "--seeds", "0", "1", "--out", str(tmp_path),
         "--results-file", "transfer_matrix.json"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr

    stats = json.loads((tmp_path / "results_seeds.json").read_text())["stats"]
    assert stats["dar -> swe"]["transfer gap"]["mean"] == 2.0
    assert stats["dar -> swe"]["protocol"]["values"] == ["cross-family dar->swe"] * 2


def test_unrecognised_results_payload_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "seed_0"
    out.mkdir()
    (out / "results.json").write_text(json.dumps({"matrix": {"dar": {}}}))
    proc = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "run_seeds.py"),
         "--aggregate-only", "--seeds", "0", "--out", str(tmp_path)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode != 0
    assert "expected a list of rows" in proc.stderr


def test_constraint_summary_separates_run_level_from_final_snapshot() -> None:
    """The bug this exists for: two runs with the same excursions, different luck."""
    from pspe.utils import constraint_summary

    limit = 0.936
    spike_early = [0.26, 0.23, 1.74, 0.22, 0.27]   # reported violation 0.0 at the end
    spike_late = [0.26, 0.23, 0.22, 0.27, 1.42]    # reported violation 1.0 at the end

    a = constraint_summary(spike_early, limit)
    b = constraint_summary(spike_late, limit)
    assert a["eval/violating_eval_fraction"] == b["eval/violating_eval_fraction"] == 0.2
    assert a["eval/cost_max_over_run"] > limit and b["eval/cost_max_over_run"] > limit

    persistent = constraint_summary([0.26, 0.42, 8.07, 8.16, 8.2], limit)
    assert persistent["eval/violating_eval_fraction"] == 0.6, "a collapse must not look transient"


def test_constraint_summary_is_empty_without_evaluations() -> None:
    from pspe.utils import constraint_summary
    assert constraint_summary([], 1.0) == {}


def test_wrong_key_is_refused_rather_than_silently_merging_rows(tmp_path: Path) -> None:
    """Rows keyed 'arm' aggregated under 'run' would average all arms together."""
    out = tmp_path / "seed_0"
    out.mkdir()
    (out / "results.json").write_text(json.dumps([
        {"arm": "pspe", "score": 1.0}, {"arm": "cnn", "score": 5.0},
    ]))
    proc = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "run_seeds.py"),
         "--aggregate-only", "--seeds", "0", "--out", str(tmp_path)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode != 0
    assert "no 'run' column" in proc.stderr and "arm" in proc.stderr


def test_key_option_aggregates_per_arm(tmp_path: Path) -> None:
    for seed, (a, b) in ((0, (1.0, 5.0)), (1, (3.0, 7.0))):
        out = tmp_path / f"seed_{seed}"
        out.mkdir()
        (out / "results.json").write_text(json.dumps([
            {"arm": "pspe", "score": a}, {"arm": "cnn", "score": b},
        ]))
    proc = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "run_seeds.py"),
         "--aggregate-only", "--seeds", "0", "1", "--out", str(tmp_path), "--key", "arm"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    stats = json.loads((tmp_path / "results_seeds.json").read_text())["stats"]
    assert stats["pspe"]["score"]["mean"] == 2.0
    assert stats["cnn"]["score"]["mean"] == 6.0
