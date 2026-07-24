"""Baselines for the controlled comparisons in Phases 1, 2 and 5.

* `operator_baselines` - FNO / DeepONet / GNOT surrogates under one training loop.
* `safe_rl`            - CPO, PID-Lagrangian PPO, Sauté RL and primal-dual NPG.

The safe-RL four are the baselines the proposal names. They are implemented
in-tree rather than imported from OmniSafe: OmniSafe does not install cleanly
against Python 3.11+/gymnasium>=0.29 at the time of writing, and pinning the
whole repo backwards to accommodate it would cost more than reimplementing four
well-specified algorithms. They share this repo's env, reward, cost and
network code, which makes the comparison tighter than a cross-library one.
"""

from . import safe_rl  # noqa: F401

__all__ = ["safe_rl"]
