"""PID-Lagrangian dual update for the CMDP constraint.

A plain Lagrangian multiplier integrates the constraint violation, which makes
the reward/cost trade-off oscillate: the multiplier keeps climbing while the
policy is still unsafe, overshoots, then decays. Stooke et al.'s PID view adds
proportional and derivative terms so the dual variable reacts to the current
violation and its trend, not only its accumulated history:

    lambda_k = max(0, K_p * e_k + K_i * I_k + K_d * max(0, e_k - e_{k-1}))
    I_k      = max(0, I_{k-1} + e_k)          # integral, kept non-negative
    e_k      = J_C(pi_k) - d                  # constraint violation

The derivative term is one-sided (only rising violation is penalised), which is
the standard anti-windup choice for this controller.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PIDLagrangian:
    """Dual controller for a single constraint `J_C <= cost_limit`."""

    cost_limit: float
    kp: float = 0.05
    ki: float = 0.0005
    kd: float = 0.02
    lambda_max: float = 50.0
    ema: float = 0.9  # smoothing on the measured cost, which is a noisy estimate

    def __post_init__(self) -> None:
        self.integral = 0.0
        self.prev_cost = 0.0
        self.smoothed_cost = 0.0
        self.multiplier = 0.0
        self._initialised = False

    def update(self, episode_cost: float) -> float:
        """Feed the measured episode cost; return the new multiplier."""
        if not self._initialised:
            self.smoothed_cost = episode_cost
            self.prev_cost = episode_cost
            self._initialised = True
        else:
            self.smoothed_cost = (
                self.ema * self.smoothed_cost + (1 - self.ema) * episode_cost
            )

        error = self.smoothed_cost - self.cost_limit
        self.integral = max(0.0, self.integral + error)
        derivative = max(0.0, self.smoothed_cost - self.prev_cost)
        self.prev_cost = self.smoothed_cost

        raw = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.multiplier = float(min(max(raw, 0.0), self.lambda_max))
        return self.multiplier

    def state(self) -> dict[str, float]:
        return {
            "lambda": self.multiplier,
            "lambda_integral": self.integral,
            "cost_smoothed": self.smoothed_cost,
            "cost_limit": self.cost_limit,
        }
