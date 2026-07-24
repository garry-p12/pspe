"""Channel-padded surrogate: one head scored across all three PDE families.

The three testbeds have different state arities (dar 1, rdf 2, swe 3), so a
surrogate built for one cannot be evaluated on another - which makes the
proposal's cross-PDE-family transfer claim unmeasurable as written.

`ChannelPaddedSurrogate` closes that: the core operator always sees
`MAX_CHANNELS` state channels plus a validity mask, with absent channels zeroed.
A model trained on `dar` (1 real channel, 2 padded) can then be rolled out on
`rdf` or `swe` without changing a weight, and the transfer gap is a number
about the dynamics rather than about tensor shapes.

The mask is not decoration. Without it the model cannot distinguish "this
channel is zero because the field is zero" from "this channel does not exist in
this family", and the padded channels would be indistinguishable from a
genuinely quiescent field.

Cost: the core carries `MAX_CHANNELS + 1` input channels instead of `C`, so a
single-family run is slightly larger than a bespoke model. `make_surrogate(...,
padded=True)` opts in; the default stays bespoke, since most runs are
single-family.
"""

from __future__ import annotations

import torch
import torch.nn as nn

Tensor = torch.Tensor

MAX_CHANNELS = 3  # max over TESTBEDS: swe has (h, u, v)


class ChannelPaddedSurrogate(nn.Module):
    """Wraps a fixed-arity operator so it accepts any testbed's channel count."""

    def __init__(self, core: nn.Module, max_channels: int = MAX_CHANNELS) -> None:
        super().__init__()
        self.core = core
        self.max_channels = max_channels

    def _pad(self, state: Tensor) -> tuple[Tensor, Tensor]:
        batch, channels, height, width = state.shape
        if channels > self.max_channels:
            raise ValueError(
                f"state has {channels} channels, more than the padded width "
                f"{self.max_channels}"
            )
        padded = state.new_zeros(batch, self.max_channels, height, width)
        padded[:, :channels] = state
        mask = state.new_zeros(batch, 1, height, width)
        mask[:] = channels / self.max_channels  # which arity this sample is
        return padded, mask

    def forward(self, state: Tensor, control: Tensor | None = None) -> Tensor:
        channels = state.shape[1]
        padded, mask = self._pad(state)
        if control is None:
            control = state.new_zeros(state.shape[0], 1, *state.shape[-2:])
        # The mask rides in on the control channel stack, so the core operator's
        # signature is untouched.
        out = self.core(padded, torch.cat([control, mask], dim=1))
        return out[:, :channels]


def make_padded_surrogate(kind: str = "fno", grid: int = 64, **kwargs: object) -> nn.Module:
    """A surrogate that can be trained on one family and rolled out on another."""
    from .operators import make_surrogate

    core = make_surrogate(
        kind, MAX_CHANNELS, grid=grid, control_channels=2, **kwargs  # control + mask
    )
    return ChannelPaddedSurrogate(core)


def supports_cross_family(model: nn.Module) -> bool:
    """True when `model` can be rolled out on a testbed of any arity."""
    return isinstance(model, ChannelPaddedSurrogate)
