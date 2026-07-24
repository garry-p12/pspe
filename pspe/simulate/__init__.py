from .solvers import (
    DiffusionAdvectionReaction,
    PDETestbed,
    ReactionDiffusionFront,
    ShallowWaterTransport,
    TESTBEDS,
    make_testbed,
)
from .fno import FNO2d, SpectralConv2d
from .multifamily import (
    MAX_CHANNELS,
    ChannelPaddedSurrogate,
    make_padded_surrogate,
    supports_cross_family,
)
from .operators import DeepONet2d, GNOT2d, make_surrogate
from .losses import (
    physics_residual_loss,
    relative_l2,
    rollout_consistency_loss,
    surrogate_loss,
)
from .dataset import (
    DataConfig,
    PDEDataset,
    dataset_path,
    ensure_dataset,
    generate_dataset,
    load_dataset,
)
from .rollout import (
    resolution_generalization,
    rollout_surrogate,
    rollout_truth,
    surrogate_fidelity,
)
from .trainer import SimulateTrainConfig, SimulateTrainer, load_surrogate

__all__ = [
    "PDETestbed",
    "DiffusionAdvectionReaction",
    "ShallowWaterTransport",
    "ReactionDiffusionFront",
    "TESTBEDS",
    "make_testbed",
    "FNO2d",
    "SpectralConv2d",
    "DeepONet2d",
    "GNOT2d",
    "make_surrogate",
    "ChannelPaddedSurrogate",
    "make_padded_surrogate",
    "supports_cross_family",
    "MAX_CHANNELS",
    "physics_residual_loss",
    "rollout_consistency_loss",
    "surrogate_loss",
    "relative_l2",
    "PDEDataset",
    "DataConfig",
    "generate_dataset",
    "load_dataset",
    "ensure_dataset",
    "dataset_path",
    "rollout_surrogate",
    "rollout_truth",
    "surrogate_fidelity",
    "resolution_generalization",
    "SimulateTrainer",
    "SimulateTrainConfig",
    "load_surrogate",
]
