"""PPO-Lagrangian for the catalog's Gymnasium environments.

`baselines/safe_rl.py` already implements four constrained-RL algorithms, but
all four speak to `BatchedFieldEnv`: they take a batched differentiable field,
a `TaskSpec`, and a `(B, K)` actuator vector. The fourteen environments in
`pspe/envs/catalog.py` are ordinary Gymnasium environments with per-step
`info["cost"]`, several of which wrap a LISFLOOD solver or a CNN rather than a
tensor field, so none of them can be handed to that code.

This is the generic version: one algorithm, any `Box`-to-`Box` environment with
a cost channel. It exists so the catalog's ladder can carry a row that was
*learned* rather than written down, which is the only row that tests whether
the hand-written incumbents are actually hard to beat.

## What is deliberate here

**Undiscounted.** `gamma` defaults to 1.0. The quantity the catalog reports is
the undiscounted sum of rewards over an episode, and every horizon in this gym
is 48 steps or fewer, so discounting would optimise a different objective than
the one being reported for no variance benefit worth having.

**The constraint is on the episode sum.** `PIDLagrangian` is fed the mean
*episode* cost of the batch against `task.cost_limit`, never a per-step mean --
same convention as `safe_rl.py` and as Safety-Gymnasium. A mean over steps
would be comparing a rate to a budget.

**Scale-free dual.** Cost limits in this gym span 0.03 (burned-area fraction)
to 1.48e6 (dollars). A PID controller with absolute gains cannot serve both, so
the error is normalised, `e = (J_C - d) / d`. This is what
`PIDLagrangian(normalize=True)` was added for.

**Both streams normalised.** Returns span O(1) to O(1e6) across these
environments too. Rewards and costs are each divided by a running estimate of
their own episode-return standard deviation before GAE, which puts the critic
targets on an O(1) scale; advantages are then standardised per batch. Only
then does a multiplier in [0, 50] mean the same thing on `HeatRetrofit-v0` as
on `CMDPToy-v0`.

**Observations are normalised per channel, and the statistics are frozen into
the checkpoint.** Depth in metres, elevation at 167 m, and dollars cannot share
a network input layer. The statistics are collected on the training split only
and shipped with the policy, so evaluating on the eval split does not touch
eval data even through a normaliser.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from pspe.plan.lagrangian import PIDLagrangian

Tensor = torch.Tensor


@dataclass
class GymPPOConfig:
    """Hyperparameters. One setting for all fourteen environments, on purpose:
    a per-environment sweep would make "a learned policy clears the bar" a
    statement about the sweep."""

    total_steps: int = 20_000
    batch_steps: int = 256       # environment steps collected per update
    min_batch_episodes: int = 8  # ... but never fewer whole episodes than this
    gamma: float = 1.0           # undiscounted; see the module docstring
    lam: float = 0.95            # GAE
    lr: float = 3.0e-4
    clip_ratio: float = 0.2
    epochs: int = 10             # inner PPO epochs per batch
    minibatch: int = 64
    entropy_coef: float = 1.0e-3
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02      # early-stop the inner epochs past this
    log_std_init: float = -0.5   # std 0.61 in the [-1, 1] action space
    # The deployed policy is the deterministic mean; the dual is fed the
    # SAMPLER's episode cost. Those are the same policy only if sigma is
    # small. In the 42-run sweep sigma never shrank -- it ended between 0.47
    # and 0.61 on thirteen of fourteen environments, which on a [-1, 1] box is
    # near-uniform noise -- so the dual spent the run regulating something
    # that was not going to be deployed. `Firebreak-v0` is the clearest case:
    # the sampler's cost was 0.088 and the deterministic policy's was 9.6e-07,
    # four orders of magnitude inside a limit the dual thought it was only
    # just meeting, and the mean collapsed onto the null action as a result.
    #
    # sigma does not shrink on its own here because `GaussianPolicy` scores
    # the log-probability of the UNCLIPPED sample: mass outside the box costs
    # nothing, so nothing pushes the spread in. Rather than change the density
    # (and with it every number already measured), sigma is annealed on a
    # fixed schedule from `log_std_init` to `log_std_final`, reaching the
    # floor at `log_std_anneal_frac` of the step budget. By the time the dual
    # has settled, the policy it is regulating is the policy that gets scored.
    log_std_final: float = -2.3   # std 0.10
    log_std_anneal_frac: float = 0.7
    hidden: int = 64
    # Dual gains, against the limit-normalised error, so one setting serves a
    # 0.03 burned-area budget and a $1.48M one.
    kp: float = 1.0
    ki: float = 0.05
    kd: float = 0.5
    lambda_max: float = 50.0
    # `PIDLagrangian`'s default smoothing is 0.9, a ~10-iteration time
    # constant tuned for the 200-iteration planner runs. A 20k-step budget here
    # is ~78 iterations, and at 0.9 the controller was still reading 0.41 while
    # the policy was spending 0.76 against a limit of 0.4 -- the multiplier sat
    # at 0.07 through an episode that was 90% over budget. 0.5 is a
    # two-iteration constant, which is the shortest that still averages out the
    # per-batch cost noise.
    dual_ema: float = 0.5
    # Train against a budget 5% tighter than the one feasibility is judged
    # against. The dual drives the policy TO the boundary -- that is what it is
    # for -- and a policy sitting exactly on the boundary is over it in about
    # half of episodes, so an undershoot target is what makes the deployed
    # (deterministic) policy feasible against the real limit. Applied
    # identically to all fourteen environments and reported with the row.
    cost_margin: float = 0.05
    seed: int = 0
    device: str = "cpu"


# --------------------------------------------------------------------------- #
# Running statistics
# --------------------------------------------------------------------------- #
class ObsNorm:
    """Per-channel (image) or per-dimension (vector) running mean and variance.

    Chan-Golub-LeVeque parallel update, so a batch costs one pass and the
    result does not depend on batch size. For an image observation the
    statistics are per channel rather than per pixel: a 5x80x80 observation has
    five physical quantities in it, not 32000 independent ones, and per-pixel
    statistics on a 26-scenario pool would be fitting the pool.
    """

    def __init__(self, shape: "tuple[int, ...]") -> None:
        self.shape = tuple(shape)
        self.image = len(self.shape) == 3
        n = self.shape[0]
        self.mean = np.zeros(n, dtype=np.float64)
        self.var = np.ones(n, dtype=np.float64)
        self.count = 0.0

    def update(self, batch: np.ndarray) -> None:
        if self.image:                       # (N, C, H, W) -> (C, N*H*W)
            x = np.moveaxis(batch, 1, 0).reshape(batch.shape[1], -1)
        else:                                # (N, d) -> (d, N)
            x = batch.T
        bm = x.mean(axis=1).astype(np.float64)
        bv = x.var(axis=1).astype(np.float64)
        bc = float(x.shape[1])
        # Seed from the first batch rather than blending it with a (0, 1)
        # prior. A prior centred on zero contributes delta^2 * count * bc / tot
        # to the variance, and `FloodDefend-v0`'s elevation channel sits at
        # 167 m: that term is not a rounding artefact, it is the estimate.
        if self.count == 0.0:
            self.mean, self.var, self.count = bm, bv, bc
            return
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        self.var = (m_a + m_b + delta**2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Normalise one observation or a stack of them. Clipped at +-10:
        `Firebreak-v0`'s observation reaches 5 on a channel whose training
        mean is near zero, and an unclipped outlier through a tanh network is
        a saturated layer."""
        std = np.sqrt(self.var + 1.0e-8)
        if self.image:
            m, s = self.mean[:, None, None], std[:, None, None]
        else:
            m, s = self.mean, std
        return np.clip((obs - m) / s, -10.0, 10.0).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count,
                "shape": self.shape}

    @classmethod
    def load(cls, state: dict) -> "ObsNorm":
        out = cls(state["shape"])
        out.mean = np.asarray(state["mean"], dtype=np.float64)
        out.var = np.asarray(state["var"], dtype=np.float64)
        out.count = float(state["count"])
        return out


class ReturnScale:
    """Running standard deviation of episode returns, for one stream.

    Divides a stream by its own scale so the critic regresses onto O(1)
    targets. Only the scale, never the mean: subtracting a mean from the reward
    changes the optimal policy on environments whose episodes differ in length,
    and `FloodCrossing-v0` episodes differ in length.
    """

    def __init__(self) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = 0.0

    def update(self, episode_totals: "list[float]") -> None:
        if not episode_totals:
            return
        x = np.asarray(episode_totals, dtype=np.float64)
        bm, bv, bc = float(x.mean()), float(x.var()), float(x.size)
        # Same reason as `ObsNorm.update`, and it bites harder here: this
        # counter grows by ~28 episodes per iteration rather than by a few
        # hundred thousand pixels, so a zero-centred prior against a cost
        # stream of 2e5 would dominate the scale for the first several
        # iterations -- the ones in which the dual is deciding what to do.
        if self.count == 0.0:
            self.mean, self.var, self.count = bm, bv, bc
            return
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        self.var = (self.var * self.count + bv * bc
                    + delta**2 * self.count * bc / tot) / tot
        self.count = tot

    @property
    def scale(self) -> float:
        return max(math.sqrt(self.var), 1.0e-6)


# --------------------------------------------------------------------------- #
# Networks
# --------------------------------------------------------------------------- #
def _encoder(shape: "tuple[int, ...]", hidden: int) -> "tuple[nn.Module, int]":
    """One architecture for every observation in this gym.

    Image observations run through two strided convolutions and an adaptive
    2x2 pool, so 8x8 (`FloodEvacuate-v0`) and 80x80 (`HeatRetrofit-v0`) reach
    the same 128-dimensional feature without a per-environment layer count.
    The pool is adaptive rather than a flatten because a flatten would make
    the parameter count a function of the grid, and `FloodOperate-v0` and
    `FloodOperateFine-v0` are the same task at two resolutions -- they should
    be the same policy class.
    """
    if len(shape) == 3:
        c = shape[0]
        return nn.Sequential(
            nn.Conv2d(c, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(),
            nn.Linear(128, hidden), nn.Tanh(),
        ), hidden
    return nn.Sequential(
        nn.Flatten(), nn.Linear(int(np.prod(shape)), hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
    ), hidden


class GaussianPolicy(nn.Module):
    """Tanh-mean Gaussian over the normalised action box.

    Every action space in this catalog is [-1, 1]^k, so the mean goes through a
    tanh and the sample is clipped. The log-probability is of the *unclipped*
    sample, which is the usual choice: it keeps the gradient at the boundary,
    where several of these policies belong (`flat_out` and `nothing` are both
    corner actions, and two of the hand-written incumbents sit on a corner).
    """

    def __init__(self, shape, act_dim: int, hidden: int, log_std_init: float):
        super().__init__()
        self.body, feat = _encoder(shape, hidden)
        self.mu = nn.Linear(feat, act_dim)
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.zeros_(self.mu.bias)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))
        #: Upper bound on `log_std`, lowered by the trainer's anneal. Kept as a
        #: buffer rather than a plain attribute so it travels with the
        #: checkpoint and `load_policy` rebuilds the same policy.
        self.register_buffer("log_std_cap",
                             torch.tensor(float(log_std_init)))

    def effective_log_std(self) -> Tensor:
        """`log_std`, capped by the anneal.

        A cap rather than an assignment: the parameter stays learnable, so the
        policy can still choose to be TIGHTER than the schedule, and the
        schedule only removes its freedom to stay loose. On `CMDPToy-v0` the
        learned spread reached 0.25 unaided, and that run should be unaffected.
        """
        return torch.minimum(self.log_std, self.log_std_cap)

    def dist(self, obs: Tensor) -> torch.distributions.Normal:
        mean = torch.tanh(self.mu(self.body(obs)))
        return torch.distributions.Normal(mean, self.effective_log_std().exp())

    def mean_action(self, obs: Tensor) -> Tensor:
        return torch.tanh(self.mu(self.body(obs)))


class Critic(nn.Module):
    def __init__(self, shape, hidden: int):
        super().__init__()
        self.body, feat = _encoder(shape, hidden)
        self.head = nn.Linear(feat, 1)

    def forward(self, obs: Tensor) -> Tensor:
        return self.head(self.body(obs)).squeeze(-1)


# --------------------------------------------------------------------------- #
# Rollout buffer
# --------------------------------------------------------------------------- #
@dataclass
class Batch:
    obs: np.ndarray
    act: np.ndarray
    logp: np.ndarray
    adv_r: np.ndarray
    adv_c: np.ndarray
    ret_r: np.ndarray
    ret_c: np.ndarray
    ep_returns: "list[float]" = field(default_factory=list)
    ep_costs: "list[float]" = field(default_factory=list)
    steps: int = 0


def _gae(rew: np.ndarray, val: np.ndarray, last_val: float,
         gamma: float, lam: float) -> "tuple[np.ndarray, np.ndarray]":
    """GAE over one episode. `last_val` is V(s_T) on truncation and 0.0 on
    termination -- a clock running out is not the same event as the episode
    being over, and thirteen of these fourteen environments truncate."""
    n = len(rew)
    adv = np.zeros(n, dtype=np.float64)
    nxt = last_val
    gae = 0.0
    for t in range(n - 1, -1, -1):
        delta = rew[t] + gamma * nxt - val[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
        nxt = val[t]
    return adv, adv + val


class GymPPOLagrangian:
    """One agent, one environment. `fit` returns the training history."""

    def __init__(self, env, cfg: GymPPOConfig) -> None:
        self.env = env
        self.cfg = cfg
        inner = env.unwrapped
        self.cost_limit = float(inner.task.cost_limit)
        obs_shape = tuple(env.observation_space.shape)
        act_dim = int(np.prod(env.action_space.shape))
        self.obs_shape, self.act_dim = obs_shape, act_dim

        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.dev = torch.device(cfg.device)
        self.pi = GaussianPolicy(obs_shape, act_dim, cfg.hidden,
                                 cfg.log_std_init).to(self.dev)
        self.vr = Critic(obs_shape, cfg.hidden).to(self.dev)
        self.vc = Critic(obs_shape, cfg.hidden).to(self.dev)
        self.opt = torch.optim.Adam(
            list(self.pi.parameters()) + list(self.vr.parameters())
            + list(self.vc.parameters()), lr=cfg.lr)

        self.obs_norm = ObsNorm(obs_shape)
        self.ret_scale = ReturnScale()
        self.cost_scale = ReturnScale()
        self.dual_target = self.cost_limit * (1.0 - cfg.cost_margin)
        self.dual = PIDLagrangian(
            cost_limit=self.dual_target, kp=cfg.kp, ki=cfg.ki, kd=cfg.kd,
            lambda_max=cfg.lambda_max, ema=cfg.dual_ema, normalize=True)
        self.steps_done = 0

    # -- acting ------------------------------------------------------------ #
    def _t(self, obs: np.ndarray) -> Tensor:
        return torch.as_tensor(self.obs_norm(obs)[None], device=self.dev)

    def act(self, obs: np.ndarray) -> "tuple[np.ndarray, float]":
        with torch.no_grad():
            d = self.pi.dist(self._t(obs))
            a = d.sample()
            logp = float(d.log_prob(a).sum())
        return a.squeeze(0).cpu().numpy(), logp

    def deterministic(self, obs: np.ndarray) -> np.ndarray:
        """The policy as it would be deployed. This is what gets scored."""
        with torch.no_grad():
            a = self.pi.mean_action(self._t(obs)).squeeze(0).cpu().numpy()
        return np.clip(a, -1.0, 1.0).astype(np.float32).reshape(
            self.env.action_space.shape)

    # -- collecting -------------------------------------------------------- #
    def collect(self) -> Batch:
        cfg = self.cfg
        obs_buf, act_buf, logp_buf = [], [], []
        adv_r, adv_c, ret_r, ret_c = [], [], [], []
        ep_returns, ep_costs = [], []
        steps = 0
        episodes = 0
        raw_obs_for_norm = []

        while steps < cfg.batch_steps or episodes < cfg.min_batch_episodes:
            seed = int(self.rng.integers(0, 2**31 - 1))
            obs, _ = self.env.reset(seed=seed)
            e_obs, e_act, e_logp, e_rew, e_cost = [], [], [], [], []
            while True:
                a, logp = self.act(obs)
                e_obs.append(np.asarray(obs, dtype=np.float32))
                e_act.append(a)
                e_logp.append(logp)
                nxt, rew, term, trunc, info = self.env.step(
                    np.clip(a, -1.0, 1.0).astype(np.float32).reshape(
                        self.env.action_space.shape))
                e_rew.append(float(rew))
                e_cost.append(float(info["cost"]))
                obs = nxt
                if term or trunc:
                    break
            n = len(e_rew)
            steps += n
            episodes += 1
            ep_returns.append(float(np.sum(e_rew)))
            ep_costs.append(float(np.sum(e_cost)))
            raw_obs_for_norm.extend(e_obs)

            # Scale AFTER the raw episode totals are recorded: the dual is fed
            # raw cost, the critics see scaled cost.
            rs, cs = self.ret_scale.scale, self.cost_scale.scale
            rew_s = np.asarray(e_rew, dtype=np.float64) / rs
            cost_s = np.asarray(e_cost, dtype=np.float64) / cs

            ob = torch.as_tensor(
                np.stack([self.obs_norm(o) for o in e_obs]), device=self.dev)
            with torch.no_grad():
                v_r = self.vr(ob).cpu().numpy().astype(np.float64)
                v_c = self.vc(ob).cpu().numpy().astype(np.float64)
                if trunc and not term:
                    last_r = float(self.vr(self._t(obs)))
                    last_c = float(self.vc(self._t(obs)))
                else:
                    last_r = last_c = 0.0
            a_r, r_r = _gae(rew_s, v_r, last_r, cfg.gamma, cfg.lam)
            a_c, r_c = _gae(cost_s, v_c, last_c, cfg.gamma, cfg.lam)

            obs_buf.extend(e_obs)
            act_buf.extend(e_act)
            logp_buf.extend(e_logp)
            adv_r.append(a_r); adv_c.append(a_c)
            ret_r.append(r_r); ret_c.append(r_c)

        # The normaliser is updated once per batch, from this batch's raw
        # observations. It therefore lags the data it normalised by one batch,
        # which is the price of not having a two-pass collector; on a running
        # estimate with thousands of samples behind it that lag is noise.
        self.obs_norm.update(np.stack(raw_obs_for_norm))
        self.ret_scale.update(ep_returns)
        self.cost_scale.update(ep_costs)
        self.steps_done += steps
        return Batch(
            obs=np.stack(obs_buf), act=np.stack(act_buf),
            logp=np.asarray(logp_buf, dtype=np.float32),
            adv_r=np.concatenate(adv_r), adv_c=np.concatenate(adv_c),
            ret_r=np.concatenate(ret_r), ret_c=np.concatenate(ret_c),
            ep_returns=ep_returns, ep_costs=ep_costs, steps=steps)

    # -- updating ---------------------------------------------------------- #
    def update(self, batch: Batch, multiplier: float) -> dict:
        cfg = self.cfg
        obs = torch.as_tensor(
            np.stack([self.obs_norm(o) for o in batch.obs]), device=self.dev)
        act = torch.as_tensor(batch.act, device=self.dev)
        logp_old = torch.as_tensor(batch.logp, device=self.dev)
        ret_r = torch.as_tensor(batch.ret_r, dtype=torch.float32, device=self.dev)
        ret_c = torch.as_tensor(batch.ret_c, dtype=torch.float32, device=self.dev)

        def std_(x: np.ndarray) -> Tensor:
            t = torch.as_tensor(x, dtype=torch.float32, device=self.dev)
            return (t - t.mean()) / (t.std() + 1.0e-8)

        a_r, a_c = std_(batch.adv_r), std_(batch.adv_c)
        # The (1 + lambda) divisor keeps the gradient magnitude roughly fixed
        # as the multiplier grows. Without it a large multiplier is also a
        # large learning rate, which is how a Lagrangian run diverges just as
        # the constraint starts to bind.
        adv = (a_r - multiplier * a_c) / (1.0 + multiplier)

        n = obs.shape[0]
        idx = np.arange(n)
        kl_last = 0.0
        stopped = 0
        for epoch in range(cfg.epochs):
            self.rng.shuffle(idx)
            for s in range(0, n, cfg.minibatch):
                mb = torch.as_tensor(idx[s:s + cfg.minibatch], device=self.dev)
                d = self.pi.dist(obs[mb])
                logp = d.log_prob(act[mb]).sum(-1)
                ratio = (logp - logp_old[mb]).exp()
                clipped = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio)
                pi_loss = -torch.min(ratio * adv[mb], clipped * adv[mb]).mean()
                ent = d.entropy().sum(-1).mean()
                v_loss = ((self.vr(obs[mb]) - ret_r[mb]) ** 2).mean() \
                    + ((self.vc(obs[mb]) - ret_c[mb]) ** 2).mean()
                loss = pi_loss - cfg.entropy_coef * ent + cfg.value_coef * v_loss
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.pi.parameters()) + list(self.vr.parameters())
                    + list(self.vc.parameters()), cfg.max_grad_norm)
                self.opt.step()
            with torch.no_grad():
                d = self.pi.dist(obs)
                kl_last = float(
                    (logp_old - d.log_prob(act).sum(-1)).mean().clamp(min=0.0))
            if kl_last > 1.5 * cfg.target_kl:
                stopped = epoch + 1
                break
        # `.detach()` on the way out, not `float()` alone: these three are
        # the last graph-carrying tensors of the epoch loop and torch warns
        # that converting one straight to a scalar "may lead to unexpected
        # behavior". They are diagnostics -- nothing downstream of this dict
        # is differentiated -- so the right fix is to drop the graph rather
        # than to silence the warning.
        return {"kl": kl_last, "epochs_run": stopped or cfg.epochs,
                "entropy": float(ent.detach()),
                "pi_loss": float(pi_loss.detach()),
                "v_loss": float(v_loss.detach())}

    # -- the loop ---------------------------------------------------------- #
    def _anneal_log_std(self) -> None:
        """Lower the spread cap linearly to `log_std_final`.

        Linear in STEPS rather than in iterations, because the fourteen
        environments differ in episode length and so in iterations per step,
        and the schedule has to mean the same thing in all of them.
        """
        cfg = self.cfg
        frac = min(1.0, self.steps_done
                   / max(cfg.log_std_anneal_frac * cfg.total_steps, 1.0))
        cap = cfg.log_std_init + frac * (cfg.log_std_final - cfg.log_std_init)
        self.pi.log_std_cap.fill_(cap)

    def fit(self, on_iteration=None) -> "list[dict]":
        history: "list[dict]" = []
        multiplier = 0.0
        it = 0
        while self.steps_done < self.cfg.total_steps:
            self._anneal_log_std()
            batch = self.collect()
            mean_cost = float(np.mean(batch.ep_costs))
            stats = self.update(batch, multiplier)
            # Dual AFTER the policy update, on the cost the policy that
            # produced the batch actually incurred.
            multiplier = self.dual.update(mean_cost)
            it += 1
            row = {
                "iteration": it, "steps": self.steps_done,
                "episodes": len(batch.ep_returns),
                "train_return": float(np.mean(batch.ep_returns)),
                "train_return_sd": float(np.std(batch.ep_returns)),
                "train_cost": mean_cost,
                "train_feasible_frac": float(np.mean(
                    np.asarray(batch.ep_costs) <= self.cost_limit + 1e-9)),
                "multiplier": float(multiplier),
                "sigma": float(self.pi.effective_log_std().detach().exp().mean()),
                **stats,
            }
            history.append(row)
            if on_iteration is not None:
                on_iteration(row, self)
        return history

    # -- checkpoint -------------------------------------------------------- #
    def state(self, extra: "dict | None" = None) -> dict:
        return {
            "policy": {k: v.cpu() for k, v in self.pi.state_dict().items()},
            "obs_norm": self.obs_norm.state(),
            "obs_shape": self.obs_shape, "act_dim": self.act_dim,
            "hidden": self.cfg.hidden, "cost_limit": self.cost_limit,
            "dual_target": self.dual_target,
            "cfg": self.cfg.__dict__, **(extra or {}),
        }


def load_policy(state: dict, action_shape, device: str = "cpu"):
    """Rebuild the deterministic policy from a checkpoint.

    Returns a callable from observation to action, which is exactly the
    interface `scripts/score_gym.py` already scores every hand-written rung
    through -- so a learned row is measured by the same code as the rest of the
    ladder, and not by a second evaluator that could disagree with it.
    """
    pi = GaussianPolicy(tuple(state["obs_shape"]), int(state["act_dim"]),
                        int(state["hidden"]), 0.0).to(device)
    pi.load_state_dict(state["policy"])
    pi.eval()
    norm = ObsNorm.load(state["obs_norm"])
    dev = torch.device(device)

    def policy(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(norm(obs)[None], device=dev)
            a = pi.mean_action(x).squeeze(0).cpu().numpy()
        return np.clip(a, -1.0, 1.0).astype(np.float32).reshape(action_shape)

    return policy
