"""A differentiable stand-in for one control interval of the flood solver.

Why this exists, stated plainly: without it, `HybridPlannerTrainer` runs on
this environment with `alpha` pinned to zero, which makes it plain
likelihood-ratio with a critic baseline. The PID dual still runs, the
saturation wall still runs — but the adaptive pathwise/likelihood mixing, the
thing the method is named for, never fires. Reporting that as "the framework
on the flood twin" would be claiming the idea and running the plumbing.

The obstacle is real and was measured, not assumed. Autograd survives
`FloodSolver`, but a control interval is ~250 adaptive CFL steps and an
episode is thousands; keeping that graph costs gigabytes, and the gate's
orifice derivative `dq/d(dH) ~ dH^(-1/2)` spans twenty-five orders of
magnitude across an episode (see the note in `flood_env.py`). So the truth
dynamics cannot supply a usable pathwise gradient at episode length. That is
exactly the situation `BatchedFieldEnv(dynamics='surrogate')` was built for,
and this is the flood half of it.

## What is learned and what is not

Only the interior evolution. The river boundary is re-held every interval by
`FloodOperateEnv._apply_river`, which is a boundary condition rather than
dynamics, and the structures' own exchange is physics with a closed form. The
surrogate predicts the CHANGE in depth over one control interval given the
depth now, the bed, the masks and the control setting — a residual, because
predicting `h` directly makes the network spend its capacity reproducing the
bed it was already given.

## What it is not

It is not a validated flood model and is not used as one. It exists to carry
a gradient, and the honest report of a planner trained on it is its score
under the TRUE solver — which is what `HybridPlannerTrainer`'s `eval_env` and
`probe_real_cost` are for, and why `measure_pathwise_bias` compares the two.
A planner that only looks good on its own surrogate has not solved anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..simulate.real.models import UNet

Tensor = torch.Tensor

#: h, bed, protected, river, gate, pump, rain — the inputs one interval needs.
IN_CHANNELS = 7


class FloodIntervalSurrogate(nn.Module):
    """(depth, terrain, control) -> change in depth over one control interval.

    Residual by construction, and non-negative by construction: depth cannot
    go below zero and a network free to predict negative water will, near the
    dry edge, where most of the cells are.
    """

    def __init__(self, width: int = 24, depth: int = 3) -> None:
        super().__init__()
        self.net = UNet(IN_CHANNELS, width=width, depth=depth)
        #: Depth changes are centimetres against a bed in metres; without a
        #: scale the head has to learn a 1e-2 output from 1e1 inputs.
        self.scale = 0.5

    def features(self, h: Tensor, z_norm: Tensor, protect: Tensor,
                 river: Tensor, gate: Tensor, pump: Tensor,
                 rain: Tensor) -> Tensor:
        b = h.shape[0]
        def bc(v: Tensor) -> Tensor:
            return v.view(b, 1, 1, 1).expand(b, 1, *h.shape[-2:])
        return torch.cat([
            h.unsqueeze(1), z_norm.unsqueeze(1),
            protect.unsqueeze(1), river.unsqueeze(1),
            bc(gate), bc(pump), bc(rain),
        ], dim=1)

    def forward(self, feats: Tensor) -> Tensor:
        """Returns the depth change, same (B, H, W) as the depth it was given."""
        return self.scale * self.net(feats)[:, 0]

    def step(self, h: Tensor, *args: Tensor) -> Tensor:
        """Next depth, clamped at zero so the state stays physical."""
        return torch.relu(h + self.forward(self.features(h, *args)))


@dataclass
class SurrogateFitReport:
    transitions: int
    train_rmse_m: float
    val_rmse_m: float
    #: RMSE of predicting "nothing changes". The surrogate has to beat it, and
    #: on a mostly-dry domain it is a surprisingly strong baseline.
    persistence_rmse_m: float
    epochs: int

    @property
    def skill(self) -> float:
        """1 - RMSE/persistence. Zero means it learned nothing useful."""
        if self.persistence_rmse_m <= 0:
            return 0.0
        return 1.0 - self.val_rmse_m / self.persistence_rmse_m


@torch.no_grad()
def collect_transitions(env, episodes: int, generator: torch.Generator | None = None
                        ) -> dict[str, Tensor]:
    """Roll the TRUE solver under random controls and record what it did.

    Random rather than on-policy, deliberately: a surrogate fitted only where
    the current policy goes is a surrogate that stops being right the moment
    the policy moves, and the pathwise gradient is a statement about the
    neighbourhood of the action, not about the action.
    """
    H, W = env.h_cells, env.w
    hs, nxt, gates, pumps, rains, zs = [], [], [], [], [], []
    s = env.reset(episodes, generator=generator)
    z_norm = s[:, 1].clone()
    for _ in range(env.horizon):
        h0 = env.h.detach().clone()
        a = torch.rand(episodes, env.action_dim, generator=generator) * 2.0 - 1.0
        u = (a.clamp(-1, 1) + 1) / 2
        s, _, _, done = env.step(a)
        hs.append(h0)
        nxt.append(env.h.detach().clone())
        gates.append(u[:, 0])
        pumps.append(u[:, 1:].mean(dim=1) if env.n_pumps else torch.zeros(episodes))
        rains.append(env._rain_ms * 3.6e6)          # back to mm/h
        zs.append(z_norm)
        if done:
            break
    return {
        "h": torch.cat(hs), "next": torch.cat(nxt), "z": torch.cat(zs),
        "gate": torch.cat(gates), "pump": torch.cat(pumps), "rain": torch.cat(rains),
        "protect": env.protect.to(torch.float32).expand(len(hs) * episodes, H, W),
        "river": env.river.to(torch.float32).expand(len(hs) * episodes, H, W),
    }


def fit_surrogate(data: dict[str, Tensor], *, epochs: int = 40, batch: int = 32,
                  lr: float = 2e-3, val_frac: float = 0.2,
                  generator: torch.Generator | None = None,
                  model: FloodIntervalSurrogate | None = None,
                  ) -> tuple[FloodIntervalSurrogate, SurrogateFitReport]:
    """Fit on recorded transitions, held out by SAMPLE rather than by episode.

    A split by sample is the easy one and is the one reported here; it answers
    "can it interpolate within the conditions it saw". It does not answer
    whether it generalises to a hydrograph it has never seen, and the planner's
    honest score is its evaluation under the true solver regardless.
    """
    net = model or FloodIntervalSurrogate()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = data["h"].shape[0]
    perm = torch.randperm(n, generator=generator)
    cut = int(n * (1 - val_frac))
    tr, va = perm[:cut], perm[cut:]
    target = data["next"] - data["h"]

    def feats(idx: Tensor) -> Tensor:
        return net.features(data["h"][idx], data["z"][idx], data["protect"][idx],
                            data["river"][idx], data["gate"][idx],
                            data["pump"][idx], data["rain"][idx])

    last = 0.0
    for _ in range(epochs):
        net.train()
        order = tr[torch.randperm(tr.numel(), generator=generator)]
        for i in range(0, order.numel(), batch):
            idx = order[i:i + batch]
            loss = (net(feats(idx)) - target[idx]).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            last = float(loss.detach())

    net.eval()
    with torch.no_grad():
        pred = torch.cat([net(feats(va[i:i + 64])) for i in range(0, va.numel(), 64)])
        val = float((pred - target[va]).pow(2).mean().sqrt())
        persistence = float(target[va].pow(2).mean().sqrt())
    return net, SurrogateFitReport(transitions=n, train_rmse_m=last ** 0.5,
                                   val_rmse_m=val,
                                   persistence_rmse_m=persistence, epochs=epochs)
