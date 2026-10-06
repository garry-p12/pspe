"""No runner may take ExplainConfig's default brief-length cap.

Defect 47: the default is 96 tokens and the reference briefs average 136, so
every arm was cut before its cost/reward/confidence footer, the parsed std fell
back to 0.4, and the F ~ 0.57 plateau was read as a learning limit rather than
as the cap it was. Lifting it took the supervised arm to 0.901 against a
reference of 0.901 -- and, separately, revealed that the faithfulness objective
costs 0.396 of F, which the truncation had been masking.

The cap is an experimental condition, so it must be passed explicitly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Every file that constructs an ExplainConfig for a reported run.
RUNNERS = [
    "eval/run_conformal.py",
    "eval/run_ndws_explain.py",
    "eval/run_explain_baselines.py",
    "eval/run_ablations.py",
    "scripts/train_e2e.py",
]


@pytest.mark.parametrize("rel", RUNNERS)
def test_runner_sets_max_len_explicitly(rel):
    src = (ROOT / rel).read_text()
    for m in re.finditer(r"ExplainConfig\(([^)]*)\)", src, re.S):
        args = m.group(1)
        assert "max_len" in args, (
            f"{rel} builds ExplainConfig({args.strip()[:60]}...) without max_len; "
            "it will silently take the 96-token default and cap F (defect 47)"
        )


def test_the_default_is_still_the_trap_this_guards():
    """If the library default ever becomes >= 136, this guard can be relaxed."""
    from pspe.explain import ExplainConfig

    assert ExplainConfig().max_len == 96, (
        "ExplainConfig.max_len is no longer 96; re-check whether the explicit "
        "max_len in every runner is still necessary, and update this test"
    )


def test_reference_briefs_need_more_than_the_default():
    """The measured fact behind the guard: 136-token references against a 96 cap."""
    from pspe.explain import ExplainConfig

    REFERENCE_MEAN_TOKENS = 136      # measured on runs/conformal_real
    assert ExplainConfig().max_len < REFERENCE_MEAN_TOKENS
    assert 160 >= REFERENCE_MEAN_TOKENS, "the chosen cap must clear the reference"
