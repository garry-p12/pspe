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
from ..utils.common import constraint_summary, peak_memory_mb, timer
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
    # --- closing the surrogate/reality gap on the constraint ----------------
    # The dual is fed the cost measured on *surrogate* rollouts, but violation
    # is realised under the true dynamics. Where the surrogate under-predicts
    # cost the policy is safe in-model and unsafe in reality: measured on dar,
    # 7.3% of evaluations exceeded a 0.936 limit while every model-free
    # baseline exceeded it on none.
    #
    # `real_cost_every` probes the true environment every N iterations and
    # feeds *that* cost to the dual. Between probes the surrogate's cost is
    # corrected by the running bias those probes measure, so the correction
    # applies at every step rather than only on probe steps.
    #
    # Off by default: switching it on changes the planner's real-sample cost,
    # which is the axis its advantage over model-free baselines is claimed on,
    # so it must be an explicit, accounted choice. Probe transitions are added
    # to `samples_real_env`.
    real_cost_every: int = 0          # 0 disables the probe entirely
    real_cost_episodes: int = 8
    # Constraint tightening: plan against `limit - k * sigma`, where sigma is
    # the running spread of the surrogate's cost error. With k = 0 the limit is
    # the nominal one; k > 0 buys safety margin at some return.
    cost_margin_k: float = 0.0
    # Eq. 8 of the paper: alpha* = V_L / (B^2 + V_p + V_L), where B is the
    # pathwise gradient bias from surrogate error. The paper leaves the constant
    # in B = C * eps unknown. Here B is *measured*: the true solver is
    # differentiable, so at each probe the pathwise gradient is taken through
    # both the surrogate and the truth from identical states, and B^2 is the
    # squared distance between them. Needs `real_cost_every > 0` for the probe
    # schedule; costs one extra differentiable truth rollout per probe.
    eq8_alpha: bool = False
    # --- saturation guard (off by default; dar never needed it) -------------
    # On rdf the reward-greedy sprint drives the pre-tanh policy mean past
    # |mu| ~ 3 within 40 iterations. There tanh'(mu) ~ 0: the pathwise
    # gradient dies (|g_pw| 0.036, variance 0), the variance rule then picks
    # the dead branch (alpha -> 0.99) because zero variance looks like
    # precision, and lambda = 38 multiplies nothing. `saturation_coef` adds
    # c * relu(|mu| - saturation_margin)^2 per actuator to the loss, a soft
    # wall that keeps the policy where the squash has slope. tanh'(1.5) = 0.18.
    saturation_coef: float = 0.0
    saturation_margin: float = 1.5
    # Advantage normalisation divides by the batch std; when every rollout
    # returns the same value (a saturated, near-deterministic policy) that std
    # is ~0 and the likelihood-ratio gradient explodes (|g_lr| 1e4 on rdf).
    # Legacy value 1e-6 reproduces earlier runs; 1e-2 is the sane floor.
    advantage_std_floor: float = 1e-6
    # Dual responsiveness. The PID smooths the measured cost with an EMA; 0.9
    # lags the cost by ~10 iterations, which on rdf is a third of the sprint.
    # `lambda_init` warm-starts the multiplier so early greedy steps are
    # already priced.
    dual_ema: float = 0.9
    lambda_init: float = 0.0
    dual_normalize: bool = False   # error in units of the limit (see PIDLagrangian)
    # Margin against the policy's own episode-to-episode cost spread, measured
    # on the probe's real rollouts, in addition to the surrogate-error spread.
    # A dual that holds the MEAN cost at the limit violates on ~half of
    # evaluations by construction; the surrogate-error sigma does not see that.
    margin_episode_std: bool = False
    # Conformal margin. `limit - k*sigma` is a reduction with no stated failure
    # rate: measured across 15 seed-runs it left ~1% of evaluations violating,
    # down from 7.3%, but nothing bounds that. Split conformal over the probe's
    # own cost errors does: with n errors and level delta, the ceil((n+1)(1-delta))-th
    # smallest error is a margin under which P(real cost > limit) <= delta,
    # assuming the errors are exchangeable across probes.
    margin_conformal: bool = False
    margin_delta: float = 0.1
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
    saturation: Tensor | None = None    # (B, T) — pre-tanh excess beyond the margin
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
            cost_limit=env.task.cost_limit, kp=self.cfg.kp, ki=self.cfg.ki, kd=self.cfg.kd,
            ema=self.cfg.dual_ema, lambda_init=self.cfg.lambda_init,
            normalize=self.cfg.dual_normalize,
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
        # Every periodic evaluation's episode cost, so the summary can report
        # constraint satisfaction over the run rather than at one lucky instant.
        self.eval_costs: list[float] = []
        # Surrogate-vs-truth cost error, learned from the real probes.
        self.cost_bias = 0.0          # EMA of (real cost - surrogate cost)
        self.cost_bias_var = 0.0      # EMA of squared deviation, for the margin
        self.probe_episode_std = 0.0  # spread of real episode costs at the last probe
        self.probe_errors: list[float] = []      # real minus surrogate cost, one per probe
        self.probe_episode_dev: list[float] = []  # real episode cost minus its probe mean
        self.real_probe_transitions = 0
        self._probe_seed = self.cfg.seed + 20_000
        self.pathwise_bias_sq = 0.0

    # -- rollout ------------------------------------------------------------ #
    def collect(self, env: BatchedFieldEnv, batch: int, keep_graph: bool = True) -> RolloutBatch:
        state = env.reset(batch, self.generator)
        rewards, costs, log_probs, entropies = [], [], [], []
        states, actions, equities, saturations = [], [], [], []

        for _ in range(self.cfg.horizon):
            action, log_prob = self.policy.sample(state)
            entropy = self.policy.entropy(state)
            if self.cfg.saturation_coef > 0:
                mu = self.policy.last_pre_tanh_mean
                saturations.append(
                    torch.relu(mu.abs() - self.cfg.saturation_margin).pow(2).sum(-1)
                )
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
            saturation=torch.stack(saturations, dim=1) if saturations else None,
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
        advantage = (advantage - advantage.mean()) / (
            advantage.std().clamp_min(self.cfg.advantage_std_floor) + 1e-6
        )
        likelihood = -(batch.log_probs * advantage).sum(dim=1)

        entropy_bonus = self.cfg.entropy_coef * batch.entropies.sum(dim=1)
        pathwise = pathwise - entropy_bonus
        likelihood = likelihood - entropy_bonus
        # The saturation wall is a direct function of the parameters, so it is
        # the same term on both branches and survives any mixing weight.
        saturation = 0.0
        if batch.saturation is not None:
            sat_pen = self.cfg.saturation_coef * batch.saturation.sum(dim=1)
            pathwise = pathwise + sat_pen
            likelihood = likelihood + sat_pen
            saturation = float(batch.saturation.sum(dim=1).mean().detach())

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
            "train/saturation": saturation,
            "train/pre_tanh_mean_abs": float(self.policy.last_pre_tanh_mean.abs().mean().detach())
            if hasattr(self.policy, "last_pre_tanh_mean") else 0.0,
        }
        return pathwise, likelihood, diagnostics

    # -- surrogate/reality cost gap ------------------------------------------ #
    @torch.no_grad()
    def probe_real_cost(self, episodes: int) -> float:
        """Mean episode cost of the current policy under the TRUE dynamics.

        Sampled stochastically, not greedily: the dual is controlling the cost
        of the behaviour policy, and the greedy action is not what the policy
        actually does during training.

        RNG is forked and the probe uses its own stream, for the same reason
        evaluation does — a probe that shifts the training draws would make the
        probe schedule part of the result.
        """
        env = self.eval_env
        if env is None:
            return float("nan")
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            gen = torch.Generator().manual_seed(self._probe_seed)
            self._probe_seed += 1
            state = env.reset(episodes, gen)
            total = torch.zeros(episodes, device=state.device)
            for _ in range(self.cfg.horizon):
                action, _ = self.policy.sample(state)
                state, _, cost, _ = env.step(action)
                total += cost
        self.real_probe_transitions += episodes * self.cfg.horizon
        self.probe_episode_std = float(total.std(unbiased=False)) if episodes > 1 else 0.0
        # Deviations of individual real episodes from their probe mean. The
        # conformal margin is a quantile of THESE, not of the surrogate's error:
        # measured on rdf the surrogate is accurate (bias -0.08) while the
        # violations come from the policy's own episode spread, so a quantile
        # over model error produced a margin of 0.25 where 0.69 was needed and
        # left 18.2% of evaluations violating.
        mean = float(total.mean())
        self.probe_episode_dev.extend(float(c) - mean for c in total)
        return mean

    def _pathwise_grad_from(self, env: BatchedFieldEnv, state0: Tensor, multiplier: float) -> Tensor:
        """Flat pathwise gradient of the Lagrangian rolled out from `state0` in `env`."""
        env.state = state0.clone()
        state = env.state
        total = torch.zeros(state.shape[0], device=state.device)
        for _ in range(self.cfg.horizon):
            action, _ = self.policy.sample(state)
            state, reward, cost, _ = env.step(action)
            total = total + (reward - multiplier * cost)
        loss = -total.mean()
        grads = torch.autograd.grad(loss, list(self.policy.parameters()), allow_unused=True)
        return torch.cat([
            (g if g is not None else torch.zeros_like(p)).reshape(-1)
            for g, p in zip(grads, self.policy.parameters())
        ]).detach()

    def measure_pathwise_bias(self, state0: Tensor, multiplier: float) -> float:
        """B^2 = ||g_pw(surrogate) - g_pw(truth)||^2 from identical start states.

        Both rollouts use the same policy noise (RNG forked and reseeded
        between them), so the difference isolates the dynamics model. RNG is
        restored afterwards so the measurement does not shift training.
        """
        if self.eval_env is None:
            return 0.0
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self._probe_seed)
            g_surr = self._pathwise_grad_from(self.env, state0, multiplier)
            torch.manual_seed(self._probe_seed)
            g_true = self._pathwise_grad_from(self.eval_env, state0, multiplier)
        self._probe_seed += 1
        self.real_probe_transitions += state0.shape[0] * self.cfg.horizon
        return float(((g_surr - g_true) ** 2).sum())

    def _dual_input(self, surrogate_cost: float, it: int) -> tuple[float, dict[str, float]]:
        """What the dual should see this iteration, and why.

        On a probe iteration that is the measured real cost. Between probes it
        is the surrogate's cost plus the running bias, so a surrogate that
        systematically under-predicts cost does not silently relax the
        constraint for the 19 iterations between probes.
        """
        if not self.cfg.real_cost_every or self.eval_env is None:
            return surrogate_cost, {}
        if it % self.cfg.real_cost_every == 0:
            real = self.probe_real_cost(self.cfg.real_cost_episodes)
            error = real - surrogate_cost
            # EMA rather than a full history: the bias is non-stationary, since
            # the policy keeps moving into parts of the state space where the
            # surrogate is differently wrong.
            self.cost_bias = 0.7 * self.cost_bias + 0.3 * error
            self.cost_bias_var = 0.7 * self.cost_bias_var + 0.3 * (error - self.cost_bias) ** 2
            self.probe_errors.append(error)
            record = {
                "dual/real_cost": real,
                "dual/surrogate_cost": surrogate_cost,
                "dual/cost_bias": self.cost_bias,
            }
            return real, record
        return surrogate_cost + self.cost_bias, {"dual/cost_bias": self.cost_bias}

    def _effective_limit(self) -> float:
        """Nominal limit, tightened by the measured spread of surrogate error."""
        if self.cfg.cost_margin_k <= 0:
            return self.env.task.cost_limit
        if self.cfg.margin_conformal:
            return max(0.0, self.env.task.cost_limit - self.conformal_margin())
        var = self.cost_bias_var
        if self.cfg.margin_episode_std:
            var = var + self.probe_episode_std ** 2
        sigma = var ** 0.5
        return max(0.0, self.env.task.cost_limit - self.cfg.cost_margin_k * sigma)

    def conformal_margin(self) -> float:
        """Split-conformal quantile of the probe's cost errors.

        Returns 0 until enough probes exist for the level to be attainable:
        ceil((n+1)(1-delta)) <= n requires n >= 1/delta - 1, below which no
        finite quantile gives the guarantee and a margin would be theatre.
        """
        # Bound P(real episode cost > limit) <= delta. The cost of an episode is
        # its probe mean plus a deviation; the dual holds the mean, so the margin
        # has to cover the upper delta-quantile of the deviations, plus any
        # systematic bias between surrogate and reality.
        dev = self.probe_episode_dev
        n = len(dev)
        need = max(2, int(round(1.0 / self.cfg.margin_delta)) - 1)
        if n < need:
            return 0.0
        rank = min(n, int(-(-(n + 1) * (1.0 - self.cfg.margin_delta) // 1)))
        quantile = max(0.0, sorted(dev)[rank - 1])
        bias = max(0.0, self.cost_bias)      # surrogate under-predicting cost
        return quantile + bias

    # -- joint-training hooks (no-ops here) ---------------------------------- #
    def _before_policy_step(self, batch: RolloutBatch, pathwise: Tensor,
                            likelihood: Tensor) -> dict[str, float]:
        return {}

    def _after_policy_step(self, it: int) -> dict[str, float]:
        return {}

    # -- training ----------------------------------------------------------- #
    def train(self) -> dict[str, float]:
        with timer() as clock:
            for it in range(1, self.cfg.iterations + 1):
                batch = self.collect(self.env, self.cfg.batch, keep_graph=True)
                episode_cost = float(batch.costs.sum(dim=1).mean().detach())
                dual_cost, dual_record = self._dual_input(episode_cost, it)
                self.dual.cost_limit = self._effective_limit()
                multiplier = self.dual.update(dual_cost)

                if (self.cfg.eq8_alpha and self.cfg.real_cost_every
                        and it % self.cfg.real_cost_every == 0):
                    b_sq = self.measure_pathwise_bias(batch.states[0], multiplier)
                    self.pathwise_bias_sq = 0.7 * self.pathwise_bias_sq + 0.3 * b_sq
                    self.estimator.bias_sq = self.pathwise_bias_sq
                    dual_record["alpha/pathwise_bias_sq"] = self.pathwise_bias_sq

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

                # Hooks for joint training. `_before_policy_step` runs while
                # the rollout graph is still alive (the estimator frees it), so
                # a subclass can take gradients of the planning loss w.r.t.
                # modules other than the policy. `_after_policy_step` runs once
                # the policy has moved, for those modules' own updates.
                extra = self._before_policy_step(batch, pathwise, likelihood)
                self.opt_policy.zero_grad(set_to_none=True)
                stats = self.estimator.step(pathwise, likelihood)
                self.opt_policy.step()
                extra.update(self._after_policy_step(it))

                record = {
                    **diagnostics,
                    **extra,
                    **dual_record,
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
                    self.eval_costs.append(metrics["episode_cost"])
                    self.logger.log(it, **{f"eval/{k}": v for k, v in metrics.items()})

        final = self.evaluate(self.cfg.eval_episodes) if self.eval_env is not None else {}
        if final:
            self.eval_costs.append(final["episode_cost"])
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
            **constraint_summary(self.eval_costs, self.env.task.cost_limit),
            "wall_clock_s": clock.seconds,
            "peak_memory_mb": peak_memory_mb(self.device),
            "samples": rollout_steps,  # kept for backwards compatibility
            "rollout_steps": rollout_steps,
            "samples_surrogate": rollout_steps if on_surrogate else 0,
            "samples_real_env": (
                # Probe transitions are real environment interaction and belong
                # in the number the sample-efficiency claim is made on.
                self.surrogate_train_transitions + self.real_probe_transitions
                if on_surrogate else rollout_steps
            ),
            "samples_real_probe": self.real_probe_transitions,
            "dual/final_cost_bias": self.cost_bias,
            "alpha/final_pathwise_bias_sq": self.pathwise_bias_sq,
            "dual/effective_limit": self._effective_limit(),
            "dynamics": self.env.dynamics,
            "final_alpha": self.estimator.alpha,
            "final_lambda": self.dual.multiplier,
            "train/pre_tanh_mean_abs": (
                self.history[-1].get("train/pre_tanh_mean_abs", 0.0) if self.history else 0.0
            ),
        }
        self.logger.log_summary(**summary)
        return summary

    # -- evaluation --------------------------------------------------------- #
    @torch.no_grad()
    def evaluate(self, episodes: int = 8, env: BatchedFieldEnv | None = None) -> dict[str, float]:
        """Report on the numerical dynamics, not the surrogate.

        Forked RNG: `env.step` draws globally, so an evaluation would otherwise
        shift every subsequent training draw and the eval schedule would become
        part of the result.
        """
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            return self._evaluate(episodes, env)

    @property
    def eval_seed(self) -> int:
        """Fixed evaluation set: same initial conditions at every evaluation."""
        return self.cfg.seed + 10_000

    @torch.no_grad()
    def _evaluate(self, episodes: int = 8, env: BatchedFieldEnv | None = None) -> dict[str, float]:
        env = env or self.eval_env or self.env
        self.policy.eval()
        state = env.reset(episodes, torch.Generator().manual_seed(self.eval_seed))
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
