"""Constrained hybrid-gradient planner.

Each iteration:

1. roll the *differentiable surrogate* forward `horizon` steps under the current
   policy, keeping the graph, and record per-step reward, cost and log-prob;
2. build two per-sample losses from that single rollout:
     * pathwise      - the Lagrangian objective itself, differentiated through
                       the surrogate;
     * likelihood    - log pi(a|s) * advantage, with the advantage detached;
3. mix their gradients with the adaptive coefficient (`HybridGradientEstimator`);
4. update the PID-Lagrangian multiplier from the measured episode cost;
5. periodically evaluate on the *numerical* dynamics, which is the honest
   report: a planner that only looks good on its own surrogate has not solved
   the problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from ..envs.pde_env import BatchedFieldEnv
from ..utils.common import peak_memory_mb, timer
from ..utils.logging import RunLogger
from .hybrid_gradient import HybridGradientEstimator
from .lagrangian import PIDLagrangian
from .policy import FieldCritic, GaussianFieldPolicy

Tensor = torch.Tensor


@dataclass
class PlannerConfig:
    iterations: int = 200
    batch: int = 16
    horizon: int = 12
    gamma: float = 0.98
    lr_policy: float = 3e-4
    lr_critic: float = 1e-3
    entropy_coef: float = 1e-3
    adaptive_alpha: bool = True
    alpha_init: float = 0.5
    alpha_fixed: float = 0.5           # used when adaptive_alpha is False
    grad_clip: float = 10.0
    # Dual gains must be scaled to the reward gradient, not the cost: lambda has
    # to reach |dR/da| / |dC/da| at the constrained optimum. The original values
    # (0.05 / 5e-4 / 0.02) could not - measured on dar, lambda peaked at 0.0475
    # while the episode cost ran to 1.34 against a 0.936 limit, exceeding it in
    # 99 of 200 iterations. The planner was effectively unconstrained and bought
    # its return advantage over the baselines by violating the budget they
    # respected. Same failure the safe-RL baselines had; see SafeRLConfig.
    kp: float = 0.5
    ki: float = 0.05
    kd: float = 0.1
    eval_every: int = 20
    eval_episodes: int = 8
    seed: int = 0
    log_dir: str = "runs/plan"


@dataclass
class RolloutBatch:
    rewards: Tensor          # (B, T)
    costs: Tensor            # (B, T)  — safety constraint g_1
    log_probs: Tensor        # (B, T)
    entropies: Tensor        # (B, T)
    states: list[Tensor]
    actions: list[Tensor]
    equity_costs: Tensor | None = None  # (B, T) — equity constraint g_2, if enabled
    extras: dict[str, float] = field(default_factory=dict)


class HybridPlannerTrainer:
    def __init__(
        self,
        env: BatchedFieldEnv,
        policy: GaussianFieldPolicy,
        critic: FieldCritic | None = None,
        cfg: PlannerConfig | None = None,
        eval_env: BatchedFieldEnv | None = None,
        logger: RunLogger | None = None,
        device: torch.device | str = "cpu",
        surrogate_train_transitions: int = 0,
    ) -> None:
        self.cfg = cfg or PlannerConfig()
        # Real environment transitions spent fitting the surrogate. This is the
        # planner's true sample cost when training on surrogate dynamics.
        self.surrogate_train_transitions = surrogate_train_transitions
        self.device = torch.device(device)
        self.env = env
        self.eval_env = eval_env
        self.policy = policy.to(self.device)
        self.critic = (critic or FieldCritic(env.obs_shape[0])).to(self.device)
        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.lr_policy)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr_critic)
        self.estimator = HybridGradientEstimator(
            self.policy.parameters(),
            adaptive=self.cfg.adaptive_alpha,
            alpha_init=self.cfg.alpha_init if self.cfg.adaptive_alpha else self.cfg.alpha_fixed,
            grad_clip=self.cfg.grad_clip,
        )
        self.dual = PIDLagrangian(
            cost_limit=env.task.cost_limit, kp=self.cfg.kp, ki=self.cfg.ki, kd=self.cfg.kd
        )
        # Second dual for the equity constraint g_2, only when the task enables it.
        self.equity_enabled = bool(getattr(env.task, "equity_enabled", False))
        self.equity_dual = (
            PIDLagrangian(cost_limit=env.task.equity_limit, kp=self.cfg.kp,
                          ki=self.cfg.ki, kd=self.cfg.kd)
            if self.equity_enabled else None
        )
        self.logger = logger or RunLogger(self.cfg.log_dir)
        self.generator = torch.Generator().manual_seed(self.cfg.seed)
        self.history: list[dict[str, float]] = []

    # -- rollout ------------------------------------------------------------ #
    def collect(self, env: BatchedFieldEnv, batch: int, keep_graph: bool = True) -> RolloutBatch:
        state = env.reset(batch, self.generator)
        rewards, costs, log_probs, entropies = [], [], [], []
        states, actions, equities = [], [], []

        for _ in range(self.cfg.horizon):
            action, log_prob = self.policy.sample(state)
            entropy = self.policy.entropy(state)
            states.append(state.detach())
            actions.append(action.detach())

            next_state, reward, cost, _ = env.step(action)
            if self.equity_enabled:
                # Differentiable through the surrogate, exactly like the safety
                # cost, so the equity constraint gets a pathwise gradient too.
                equities.append(env.task.equity_cost(next_state, action))
            if not keep_graph:
                next_state = next_state.detach()
                env.state = next_state
            rewards.append(reward)
            costs.append(cost)
            log_probs.append(log_prob)
            entropies.append(entropy)
            state = next_state

        return RolloutBatch(
            rewards=torch.stack(rewards, dim=1),
            costs=torch.stack(costs, dim=1),
            log_probs=torch.stack(log_probs, dim=1),
            entropies=torch.stack(entropies, dim=1),
            states=states,
            actions=actions,
            equity_costs=torch.stack(equities, dim=1) if self.equity_enabled else None,
        )

    # -- losses ------------------------------------------------------------- #
    def _discounted_to_go(self, values: Tensor) -> Tensor:
        """Reward-to-go with discount `gamma`, shape preserved (B, T)."""
        out = torch.zeros_like(values)
        running = torch.zeros_like(values[:, 0])
        for t in reversed(range(values.shape[1])):
            running = values[:, t] + self.cfg.gamma * running
            out[:, t] = running
        return out

    def _per_sample_losses(
        self, batch: RolloutBatch, multiplier: float, equity_multiplier: float = 0.0
    ) -> tuple[Tensor, Tensor, dict[str, float]]:
        """Return (pathwise_loss, likelihood_loss) per sample, plus diagnostics."""
        lagrangian_step = batch.rewards - multiplier * batch.costs
        if batch.equity_costs is not None:
            lagrangian_step = lagrangian_step - equity_multiplier * batch.equity_costs

        # --- pathwise: differentiate the objective through the surrogate.
        pathwise = -(lagrangian_step.sum(dim=1))

        # --- likelihood ratio: score function with a critic baseline.
        with torch.no_grad():
            returns = self._discounted_to_go(lagrangian_step.detach())
        flat_states = torch.cat(batch.states, dim=0)
        value, cost_value = self.critic(flat_states)
        value = value.view(self.cfg.horizon, -1).transpose(0, 1)
        advantage = returns - value.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
        likelihood = -(batch.log_probs * advantage).sum(dim=1)

        entropy_bonus = self.cfg.entropy_coef * batch.entropies.sum(dim=1)
        pathwise = pathwise - entropy_bonus
        likelihood = likelihood - entropy_bonus

        # --- critic regression (reward and cost value heads).
        with torch.no_grad():
            cost_returns = self._discounted_to_go(batch.costs.detach())
        cost_value = cost_value.view(self.cfg.horizon, -1).transpose(0, 1)
        critic_loss = ((value - returns) ** 2).mean() + ((cost_value - cost_returns) ** 2).mean()
        self.opt_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.opt_critic.step()

        diagnostics = {
            "critic/loss": float(critic_loss.detach()),
            "train/return": float(batch.rewards.sum(dim=1).mean().detach()),
            "train/episode_cost": float(batch.costs.sum(dim=1).mean().detach()),
            "train/entropy": float(batch.entropies.mean().detach()),
        }
        return pathwise, likelihood, diagnostics

    # -- training ----------------------------------------------------------- #
    def train(self) -> dict[str, float]:
        with timer() as clock:
            for it in range(1, self.cfg.iterations + 1):
                batch = self.collect(self.env, self.cfg.batch, keep_graph=True)
                episode_cost = float(batch.costs.sum(dim=1).mean().detach())
                multiplier = self.dual.update(episode_cost)

                equity_multiplier = 0.0
                equity_record: dict[str, float] = {}
                if self.equity_dual is not None and batch.equity_costs is not None:
                    episode_equity = float(batch.equity_costs.sum(dim=1).mean().detach())
                    equity_multiplier = self.equity_dual.update(episode_equity)
                    equity_record = {
                        "equity/lambda": equity_multiplier,
                        "equity/episode_cost": episode_equity,
                        "equity/limit": self.equity_dual.cost_limit,
                    }

                pathwise, likelihood, diagnostics = self._per_sample_losses(
                    batch, multiplier, equity_multiplier
                )

                self.opt_policy.zero_grad(set_to_none=True)
                stats = self.estimator.step(pathwise, likelihood)
                self.opt_policy.step()

                record = {
                    **diagnostics,
                    **self.dual.state(),
                    **equity_record,
                    "hybrid/alpha": stats.alpha,
                    "hybrid/grad_norm_pathwise": stats.grad_norm_pathwise,
                    "hybrid/grad_norm_likelihood": stats.grad_norm_likelihood,
                    "hybrid/var_pathwise": stats.var_pathwise,
                    "hybrid/var_likelihood": stats.var_likelihood,
                }
                self.logger.log(it, **record)
                self.history.append({"iteration": it, **record})

                if self.eval_env is not None and it % self.cfg.eval_every == 0:
                    metrics = self.evaluate(self.cfg.eval_episodes)
                    self.logger.log(it, **{f"eval/{k}": v for k, v in metrics.items()})

        final = self.evaluate(self.cfg.eval_episodes) if self.eval_env is not None else {}
        rollout_steps = self.cfg.iterations * self.cfg.batch * self.cfg.horizon
        # Sample accounting has to distinguish surrogate rollouts from real
        # environment interaction, or the model-based planner looks like it
        # spends the same budget as the model-free baselines when it does not.
        # Real cost = the transitions used to fit the surrogate (set by the
        # caller via `surrogate_train_transitions`), not the rollouts taken
        # inside it.
        on_surrogate = self.env.dynamics == "surrogate"
        summary = {
            **final,
            "wall_clock_s": clock.seconds,
            "peak_memory_mb": peak_memory_mb(self.device),
            "samples": rollout_steps,  # kept for backwards compatibility
            "rollout_steps": rollout_steps,
            "samples_surrogate": rollout_steps if on_surrogate else 0,
            "samples_real_env": (
                self.surrogate_train_transitions if on_surrogate else rollout_steps
            ),
            "dynamics": self.env.dynamics,
            "final_alpha": self.estimator.alpha,
            "final_lambda": self.dual.multiplier,
        }
        self.logger.log_summary(**summary)
        return summary

    # -- evaluation --------------------------------------------------------- #
    @torch.no_grad()
    def evaluate(self, episodes: int = 8, env: BatchedFieldEnv | None = None) -> dict[str, float]:
        """Report on the numerical dynamics, not the surrogate."""
        env = env or self.eval_env or self.env
        self.policy.eval()
        state = env.reset(episodes, self.generator)
        total_reward = torch.zeros(episodes, device=state.device)
        total_cost = torch.zeros(episodes, device=state.device)
        total_equity = torch.zeros(episodes, device=state.device)

        for _ in range(self.cfg.horizon):
            action, _ = self.policy.sample(state, deterministic=True)
            state, reward, cost, _ = env.step(action)
            if self.equity_enabled:
                total_equity += env.task.equity_cost(state, action)
            total_reward += reward
            total_cost += cost

        self.policy.train()
        limit = env.task.cost_limit
        metrics = {
            "return": float(total_reward.mean()),
            "episode_cost": float(total_cost.mean()),
            "cost_limit": limit,
            "violation_rate": float((total_cost > limit).float().mean()),
            "overshoot": float(torch.relu(total_cost - limit).mean()),
        }
        if self.equity_enabled:
            eq_limit = env.task.equity_limit
            metrics.update({
                "equity_cost": float(total_equity.mean()),
                "equity_limit": eq_limit,
                "equity_violation_rate": float((total_equity > eq_limit).float().mean()),
            })
        return metrics

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "critic": self.critic.state_dict(),
                "config": self.cfg.__dict__,
                "alpha": self.estimator.alpha,
                "lambda": self.dual.multiplier,
            },
            path,
        )
