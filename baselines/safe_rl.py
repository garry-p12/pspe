"""Constrained-RL baselines: CPO, PID-Lagrangian PPO, Sauté RL, primal-dual NPG.

All four are model-free: they see the environment through detached transitions,
so none of them uses the differentiable surrogate. That is exactly the axis the
proposal's planner is meant to win on (sample efficiency), and keeping every
other component identical - same env, same reward and cost functionals, same
policy architecture, same horizon - is what makes the comparison controlled.

Shared skeleton (`OnPolicySafeAgent`):
    collect on-policy rollouts -> GAE for reward and cost -> algorithm-specific
    policy update -> critic regression.

Algorithm-specific pieces:
    PPOLagrangian   clipped surrogate on (A_r - lambda * A_c), PID dual ascent
    CPO             trust-region step with a linearised cost constraint, solved
                    in the dual of the quadratically-constrained problem, with
                    the standard infeasible-case recovery step
    SauteRL         safety budget folded into the state; unsafe states are
                    absorbing with reward 0; then unconstrained PPO
    PrimalDualNPG   natural-gradient ascent on the Lagrangian, dual ascent on
                    the multiplier
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from pspe.envs.pde_env import BatchedFieldEnv
from pspe.plan.lagrangian import PIDLagrangian
from pspe.plan.policy import FieldCritic, GaussianFieldPolicy
from pspe.utils.common import peak_memory_mb, timer
from pspe.utils.logging import RunLogger

Tensor = torch.Tensor


@dataclass
class SafeRLConfig:
    iterations: int = 200
    batch: int = 16
    horizon: int = 12
    gamma: float = 0.98
    lam: float = 0.95              # GAE lambda
    lr_policy: float = 3e-4
    lr_critic: float = 1e-3
    clip_ratio: float = 0.2
    train_iters: int = 4           # inner epochs per batch (PPO-style algorithms)
    entropy_coef: float = 1e-3
    target_kl: float = 0.01        # trust-region radius for CPO / NPG
    cg_iters: int = 10
    damping: float = 0.1
    # Dual gains for PPO-Lagrangian. These must be scaled to the *reward*
    # gradient, not to the cost: the multiplier has to reach lambda* =
    # |dR/da| / |dC/da| at the constrained optimum, and gains that cannot get
    # there in the iteration budget silently degrade the algorithm to
    # unconstrained PPO. Validated against a closed-form CMDP in
    # tests/test_safe_rl_correctness.py.
    lagrangian_kp: float = 0.5
    lagrangian_ki: float = 0.05
    lagrangian_kd: float = 0.1
    backtrack_steps: int = 10
    backtrack_coef: float = 0.8
    eval_episodes: int = 8
    seed: int = 0
    log_dir: str = "runs/baselines"


# --------------------------------------------------------------------------- #
# Shared machinery
# --------------------------------------------------------------------------- #
@dataclass
class Buffer:
    states: Tensor        # (B*T, C, H, W)
    actions: Tensor       # (B*T, K)
    log_probs: Tensor     # (B*T,)
    adv_reward: Tensor    # (B*T,)
    adv_cost: Tensor      # (B*T,)
    ret_reward: Tensor    # (B*T,)
    ret_cost: Tensor      # (B*T,)
    episode_reward: float
    episode_cost: float


def _flat(params: list[nn.Parameter]) -> Tensor:
    return torch.cat([p.detach().reshape(-1) for p in params])


def _flat_grad(loss: Tensor, params: list[nn.Parameter], retain: bool = True,
               create: bool = False) -> Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=retain,
                                create_graph=create, allow_unused=True)
    return torch.cat([
        torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
        for p, g in zip(params, grads)
    ])


def _set_flat(params: list[nn.Parameter], flat: Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            p.copy_(flat[offset : offset + n].view_as(p))
            offset += n


def _conjugate_gradient(
    hvp: Callable[[Tensor], Tensor], b: Tensor, iters: int, tol: float = 1e-10
) -> Tensor:
    """Solve H x = b without forming H."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rr = r @ r
    for _ in range(iters):
        hp = hvp(p)
        alpha = rr / (p @ hp + 1e-8)
        x = x + alpha * p
        r = r - alpha * hp
        rr_new = r @ r
        if rr_new < tol:
            break
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x


class OnPolicySafeAgent:
    """Base class: rollout collection, GAE, critic fitting, evaluation."""

    name = "base"

    def __init__(
        self,
        env: BatchedFieldEnv,
        cfg: SafeRLConfig | None = None,
        policy: GaussianFieldPolicy | None = None,
        logger: RunLogger | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg or SafeRLConfig()
        self.device = torch.device(device)
        self.env = env
        in_channels = env.obs_shape[0] + (1 if self.name == "saute" else 0)
        self.policy = (policy or GaussianFieldPolicy(in_channels, env.action_dim)).to(self.device)
        self.critic = FieldCritic(in_channels).to(self.device)
        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.lr_policy)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr_critic)
        self.logger = logger or RunLogger(f"{self.cfg.log_dir}/{self.name}")
        self.generator = torch.Generator().manual_seed(self.cfg.seed)
        self.cost_limit = env.task.cost_limit
        self.samples_used = 0

    # -- observation hook (Sauté overrides it) ------------------------------ #
    def observe(self, state: Tensor, budget: Tensor) -> Tensor:
        return state

    # -- rollout ------------------------------------------------------------ #
    @torch.no_grad()
    def collect(self) -> Buffer:
        cfg = self.cfg
        state = self.env.reset(cfg.batch, self.generator)
        budget = torch.full((cfg.batch,), self.cost_limit, device=state.device)

        obs_l, act_l, logp_l, rew_l, cost_l, val_l, cval_l = [], [], [], [], [], [], []
        for _ in range(cfg.horizon):
            obs = self.observe(state, budget)
            action, log_prob = self.policy.sample(obs)
            value, cost_value = self.critic(obs)

            next_state, reward, cost, _ = self.env.step(action)
            next_state = next_state.detach()
            self.env.state = next_state

            reward, cost = self.shape(reward, cost, budget)
            budget = budget - cost

            obs_l.append(obs)
            act_l.append(action)
            logp_l.append(log_prob)
            rew_l.append(reward)
            cost_l.append(cost)
            val_l.append(value)
            cval_l.append(cost_value)
            state = next_state

        self.samples_used += cfg.batch * cfg.horizon
        rewards = torch.stack(rew_l, dim=1)
        costs = torch.stack(cost_l, dim=1)
        values = torch.stack(val_l, dim=1)
        cost_values = torch.stack(cval_l, dim=1)

        adv_r, ret_r = self._gae(rewards, values)
        adv_c, ret_c = self._gae(costs, cost_values)

        # (B, T) -> flat index t * B + b, matching `torch.cat(obs_l, dim=0)`.
        def merge(x: Tensor) -> Tensor:
            return x.transpose(0, 1).reshape(-1)

        return Buffer(
            states=torch.cat(obs_l, dim=0),
            actions=torch.cat(act_l, dim=0),
            log_probs=torch.cat(logp_l, dim=0),
            adv_reward=merge(adv_r),
            adv_cost=merge(adv_c),
            ret_reward=merge(ret_r),
            ret_cost=merge(ret_c),
            episode_reward=float(rewards.sum(dim=1).mean()),
            episode_cost=float(costs.sum(dim=1).mean()),
        )

    def shape(self, reward: Tensor, cost: Tensor, budget: Tensor) -> tuple[Tensor, Tensor]:
        """Hook for reward/cost shaping (Sauté RL uses it)."""
        return reward, cost

    def _gae(self, rewards: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        cfg = self.cfg
        adv = torch.zeros_like(rewards)
        running = torch.zeros_like(rewards[:, 0])
        next_value = torch.zeros_like(rewards[:, 0])
        for t in reversed(range(rewards.shape[1])):
            delta = rewards[:, t] + cfg.gamma * next_value - values[:, t]
            running = delta + cfg.gamma * cfg.lam * running
            adv[:, t] = running
            next_value = values[:, t]
        return adv, adv + values

    # -- shared updates ----------------------------------------------------- #
    def fit_critic(self, buf: Buffer) -> float:
        losses = []
        for _ in range(self.cfg.train_iters):
            value, cost_value = self.critic(buf.states)
            loss = ((value - buf.ret_reward) ** 2).mean() + (
                (cost_value - buf.ret_cost) ** 2
            ).mean()
            self.opt_critic.zero_grad(set_to_none=True)
            loss.backward()
            self.opt_critic.step()
            losses.append(float(loss.detach()))
        return sum(losses) / len(losses)

    def update(self, buf: Buffer) -> dict[str, float]:
        raise NotImplementedError

    # -- driver ------------------------------------------------------------- #
    def train(self) -> dict[str, float]:
        with timer() as clock:
            for it in range(1, self.cfg.iterations + 1):
                buf = self.collect()
                stats = self.update(buf)
                stats["critic/loss"] = self.fit_critic(buf)
                stats.update(
                    {
                        "train/return": buf.episode_reward,
                        "train/episode_cost": buf.episode_cost,
                        "train/samples": float(self.samples_used),
                    }
                )
                self.logger.log(it, **stats)

        metrics = self.evaluate(self.cfg.eval_episodes)
        summary = {
            **metrics,
            "algorithm": self.name,
            "wall_clock_s": clock.seconds,
            "peak_memory_mb": peak_memory_mb(self.device),
            "samples": float(self.samples_used),
        }
        self.logger.log_summary(**summary)
        return summary

    @torch.no_grad()
    def evaluate(self, episodes: int = 8) -> dict[str, float]:
        state = self.env.reset(episodes, self.generator)
        budget = torch.full((episodes,), self.cost_limit, device=state.device)
        total_reward = torch.zeros(episodes, device=state.device)
        total_cost = torch.zeros(episodes, device=state.device)
        for _ in range(self.cfg.horizon):
            action, _ = self.policy.sample(self.observe(state, budget), deterministic=True)
            state, reward, cost, _ = self.env.step(action)
            state = state.detach()
            self.env.state = state
            budget = budget - cost
            total_reward += reward
            total_cost += cost
        return {
            "return": float(total_reward.mean()),
            "episode_cost": float(total_cost.mean()),
            "cost_limit": self.cost_limit,
            "violation_rate": float((total_cost > self.cost_limit).float().mean()),
            "overshoot": float(torch.relu(total_cost - self.cost_limit).mean()),
        }


# --------------------------------------------------------------------------- #
# 1. PID-Lagrangian PPO
# --------------------------------------------------------------------------- #
class PPOLagrangian(OnPolicySafeAgent):
    name = "ppo_lagrangian"

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.dual = PIDLagrangian(
            cost_limit=self.cost_limit,
            kp=self.cfg.lagrangian_kp,
            ki=self.cfg.lagrangian_ki,
            kd=self.cfg.lagrangian_kd,
        )

    def update(self, buf: Buffer) -> dict[str, float]:
        multiplier = self.dual.update(buf.episode_cost)

        # Normalise the reward and cost advantages *separately*, then blend.
        # Normalising the combined advantage instead rescales it to unit
        # variance whatever lambda is, which throws away the multiplier's
        # magnitude - the dual controller then chases a target it cannot move,
        # and the policy oscillates between the action bounds. Dividing by
        # (1 + lambda) keeps the effective step size bounded as lambda grows.
        adv_r = (buf.adv_reward - buf.adv_reward.mean()) / (buf.adv_reward.std() + 1e-6)
        adv_c = (buf.adv_cost - buf.adv_cost.mean()) / (buf.adv_cost.std() + 1e-6)
        adv = (adv_r - multiplier * adv_c) / (1.0 + multiplier)

        approx_kl = 0.0
        for _ in range(self.cfg.train_iters):
            log_prob = self.policy.log_prob_of(buf.states, buf.actions)
            ratio = (log_prob - buf.log_probs).exp()
            clipped = ratio.clamp(1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio)
            loss = -torch.min(ratio * adv, clipped * adv).mean()
            loss = loss - self.cfg.entropy_coef * self.policy.entropy(buf.states).mean()
            self.opt_policy.zero_grad(set_to_none=True)
            loss.backward()
            self.opt_policy.step()
            approx_kl = float((buf.log_probs - log_prob).mean().detach())
            if approx_kl > 1.5 * self.cfg.target_kl:
                break
        return {"policy/kl": approx_kl, **self.dual.state()}


# --------------------------------------------------------------------------- #
# 2. CPO
# --------------------------------------------------------------------------- #
class CPO(OnPolicySafeAgent):
    """Constrained Policy Optimization (Achiam et al., 2017), compact form."""

    name = "cpo"

    def _kl(self, buf: Buffer, old_mean: Tensor, old_std: Tensor) -> Tensor:
        dist = self.policy.distribution(buf.states)
        return torch.distributions.kl_divergence(
            torch.distributions.Normal(old_mean, old_std), dist
        ).sum(-1).mean()

    def update(self, buf: Buffer) -> dict[str, float]:
        params = [p for p in self.policy.parameters() if p.requires_grad]
        with torch.no_grad():
            old = self.policy.distribution(buf.states)
            old_mean, old_std = old.mean.clone(), old.stddev.clone()

        adv_r = (buf.adv_reward - buf.adv_reward.mean()) / (buf.adv_reward.std() + 1e-6)
        # The cost advantage is centred but NOT divided by its standard
        # deviation. CPO's dual solution mixes the cost gradient `b` with the
        # constraint surplus `c = J_C - d`, which is in raw cost units; scaling
        # `b` to unit variance leaves the two in different units and the step
        # settles at a biased point that never reaches the limit. Measured on
        # the closed-form CMDP: scaling gave a persistent 0.11 action bias that
        # did not shrink from 300 to 900 iterations.
        adv_c = buf.adv_cost - buf.adv_cost.mean()

        def surrogate() -> tuple[Tensor, Tensor]:
            log_prob = self.policy.log_prob_of(buf.states, buf.actions)
            ratio = (log_prob - buf.log_probs).exp()
            return (ratio * adv_r).mean(), (ratio * adv_c).mean()

        obj, cost_obj = surrogate()
        g = _flat_grad(obj, params)                    # reward gradient
        b = _flat_grad(cost_obj, params)               # cost gradient
        obj_value, cost_obj_value = float(obj.detach()), float(cost_obj.detach())

        kl = self._kl(buf, old_mean, old_std)
        kl_grad = _flat_grad(kl, params, create=True)

        def hvp(v: Tensor) -> Tensor:
            return _flat_grad((kl_grad * v).sum(), params) + self.cfg.damping * v

        # Constraint surplus, in *per-step* units.
        #
        # `cost_obj` is a mean over timesteps, so the linearised constraint it
        # participates in is per-step; `episode_cost - cost_limit` is a sum over
        # the horizon. Mixing the two makes the dual solution's trade-off point
        # scale with the horizon, which is why CPO settled at a biased action in
        # both the scaled and unscaled cost-advantage variants. Dividing by the
        # horizon is the rescaling openai/safety-starter-agents applies for the
        # same reason.
        c = (buf.episode_cost - self.cost_limit) / max(self.cfg.horizon, 1)

        v = _conjugate_gradient(hvp, g, self.cfg.cg_iters)
        q = float(g @ v)
        recovery = False

        # Pure TRPO step, used when the cost gradient vanishes or when the
        # constraint is slack enough that this step does not breach it.
        trpo_scale = (2 * self.cfg.target_kl / max(q, 1e-8)) ** 0.5
        trpo_step = trpo_scale * v

        if float(b @ b) < 1e-8:
            step = trpo_step
        elif c < 0 and float(b @ trpo_step) <= -c:
            # Constraint slack and the unconstrained step keeps it satisfied, so
            # the CPO problem reduces to TRPO and nu = 0.
            #
            # Without this branch the dual formula is evaluated at a comfortably
            # feasible point, where c^2/s exceeds 2*delta, the denominator
            # clamps at 1e-8, lambda explodes and the step collapses to nothing:
            # CPO freezes exactly when it is safe and should be improving the
            # reward. Measured on the closed-form CMDP, it crawled from 0.082
            # toward the 0.4 optimum at ~0.02 per 300 iterations.
            step = trpo_step
        else:
            w = _conjugate_gradient(hvp, b, self.cfg.cg_iters)
            r = float(g @ w)
            s = max(float(b @ w), 1e-8)
            # Feasibility of the *linearised* problem inside the trust region.
            denominator = 2 * self.cfg.target_kl - c**2 / s

            if c > 0 and denominator <= 0:
                # No point in the trust region satisfies the linearised
                # constraint: take the pure recovery step toward feasibility.
                recovery = True
                step = -((2 * self.cfg.target_kl / s) ** 0.5) * w
            elif denominator <= 0:
                # Same degenerate denominator but the constraint is already
                # slack (c < 0), so there is nothing to recover from - the
                # trust-region step is the right move. Falling through to the
                # dual formula here would take the square root of a negative
                # number, which is the crash this branch replaces.
                step = trpo_step
            else:
                # Dual solution of the quadratically-constrained problem.
                # q - r^2/s is non-negative in exact arithmetic (Cauchy-Schwarz
                # in the H^-1 inner product, q*s >= r^2), but conjugate gradient
                # solves H^-1 only approximately, so it can come out slightly
                # negative and make the square root complex. Clamp at zero.
                numerator = max(q - r**2 / s, 0.0)
                lam = max((numerator / denominator) ** 0.5, 1e-8)
                nu = max((lam * c - r) / s, 0.0)
                step = (v - nu * w) / lam

        # Backtracking line search on the true (non-linearised) objective.
        old_params = _flat(params)
        accepted = 0
        for i in range(self.cfg.backtrack_steps):
            _set_flat(params, old_params + (self.cfg.backtrack_coef**i) * step)
            with torch.no_grad():
                new_obj, new_cost = surrogate()
                new_kl = float(self._kl(buf, old_mean, old_std))
            improves = recovery or float(new_obj) > obj_value
            # Achiam et al.'s acceptance test: the surrogate cost may only grow
            # by the slack the current policy still has.
            feasible = c <= 0 or float(new_cost) - cost_obj_value <= max(-c, 0.0)
            if new_kl <= self.cfg.target_kl and improves and feasible:
                accepted = i + 1
                break
        else:
            _set_flat(params, old_params)  # reject the step entirely

        return {
            "policy/kl": float(new_kl),
            "cpo/backtracks": float(accepted),
            "cpo/recovery": float(recovery),
            "cpo/constraint_surplus": float(c),
        }


# --------------------------------------------------------------------------- #
# 3. Sauté RL
# --------------------------------------------------------------------------- #
class SauteRL(OnPolicySafeAgent):
    """Safety budget folded into the state; unsafe = absorbing, reward 0.

    "Sauteed" MDP: the remaining budget z_{t+1} = z_t - c_t is appended to the
    observation as a constant channel, and once z < 0 the agent collects no more
    reward. The constraint becomes a property of the state, so an ordinary
    unconstrained PPO solves it.

    **The zero-reward absorbing state only penalises the agent if rewards are
    non-negative.** Every PSPE task reward is <= 0 (negated tracking error plus
    actuation effort), so an unshifted implementation makes blowing the budget
    the single most rewarding thing the agent can do - it runs straight to the
    unconstrained optimum while looking like it is doing safe RL. The rewards
    are therefore shifted by a running floor before the absorbing rule is
    applied, which is the assumption Sauté RL is stated under.

    Caught by `tests/test_safe_rl_correctness.py`, which pins the algorithm to a
    CMDP whose constrained optimum is known in closed form; the unshifted
    version converged to the unconstrained optimum on it.
    """

    name = "saute"

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.reward_floor = float("inf")

    def observe(self, state: Tensor, budget: Tensor) -> Tensor:
        normalised = (budget / max(self.cost_limit, 1e-6)).view(-1, 1, 1, 1)
        channel = normalised.expand(-1, 1, state.shape[-2], state.shape[-1])
        return torch.cat([state, channel], dim=1)

    def shape(self, reward: Tensor, cost: Tensor, budget: Tensor) -> tuple[Tensor, Tensor]:
        # Running floor, so the shift is monotone and never re-ranks past data.
        self.reward_floor = min(self.reward_floor, float(reward.min()))
        shifted = reward - self.reward_floor
        unsafe = (budget - cost) < 0
        # Cost is passed through unshifted: it is the quantity the constraint
        # and every reported metric are defined on.
        return torch.where(unsafe, torch.zeros_like(shifted), shifted), cost

    def update(self, buf: Buffer) -> dict[str, float]:
        adv = (buf.adv_reward - buf.adv_reward.mean()) / (buf.adv_reward.std() + 1e-6)
        approx_kl = 0.0
        for _ in range(self.cfg.train_iters):
            log_prob = self.policy.log_prob_of(buf.states, buf.actions)
            ratio = (log_prob - buf.log_probs).exp()
            clipped = ratio.clamp(1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio)
            loss = -torch.min(ratio * adv, clipped * adv).mean()
            loss = loss - self.cfg.entropy_coef * self.policy.entropy(buf.states).mean()
            self.opt_policy.zero_grad(set_to_none=True)
            loss.backward()
            self.opt_policy.step()
            approx_kl = float((buf.log_probs - log_prob).mean().detach())
            if approx_kl > 1.5 * self.cfg.target_kl:
                break
        return {"policy/kl": approx_kl}


# --------------------------------------------------------------------------- #
# 4. Primal-dual natural policy gradient
# --------------------------------------------------------------------------- #
class PrimalDualNPG(OnPolicySafeAgent):
    name = "primal_dual_npg"

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.multiplier = 0.0
        self.dual_lr = 0.02

    def update(self, buf: Buffer) -> dict[str, float]:
        params = [p for p in self.policy.parameters() if p.requires_grad]
        with torch.no_grad():
            old = self.policy.distribution(buf.states)
            old_mean, old_std = old.mean.clone(), old.stddev.clone()

        adv = buf.adv_reward - self.multiplier * buf.adv_cost
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        log_prob = self.policy.log_prob_of(buf.states, buf.actions)
        obj = ((log_prob - buf.log_probs).exp() * adv).mean()
        g = _flat_grad(obj, params)

        dist = self.policy.distribution(buf.states)
        kl = torch.distributions.kl_divergence(
            torch.distributions.Normal(old_mean, old_std), dist
        ).sum(-1).mean()
        kl_grad = _flat_grad(kl, params, create=True)

        def hvp(v: Tensor) -> Tensor:
            return _flat_grad((kl_grad * v).sum(), params) + self.cfg.damping * v

        nat_grad = _conjugate_gradient(hvp, g, self.cfg.cg_iters)
        scale = (2 * self.cfg.target_kl / max(float(g @ nat_grad), 1e-8)) ** 0.5
        _set_flat(params, _flat(params) + scale * nat_grad)

        # Dual ascent (projected onto the non-negative orthant).
        self.multiplier = max(
            0.0, self.multiplier + self.dual_lr * (buf.episode_cost - self.cost_limit)
        )
        return {"lambda": self.multiplier, "policy/kl": float(kl.detach())}


ALGORITHMS: dict[str, type[OnPolicySafeAgent]] = {
    "ppo_lagrangian": PPOLagrangian,
    "cpo": CPO,
    "saute": SauteRL,
    "primal_dual_npg": PrimalDualNPG,
}


def make_agent(name: str, env: BatchedFieldEnv, **kwargs: object) -> OnPolicySafeAgent:
    if name not in ALGORITHMS:
        raise KeyError(f"unknown algorithm {name!r}; expected one of {sorted(ALGORITHMS)}")
    return ALGORITHMS[name](env, **kwargs)  # type: ignore[arg-type]
