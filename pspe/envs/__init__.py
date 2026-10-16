from .actuators import GaussianActuatorBasis
from .catalog import BY_ID, SPECS, EnvSpec, catalog, describe, register_all
from .evacuate import EvacuateControlEnv
from .fire_env import FirebreakEnv
from .flood_env import (
    RIVER_BED_M, SUMP_M, FloodControlEnv, FloodOperateEnv, FloodOperateTask,
    make_polder_bank,
)
from .pde_env import BatchedFieldEnv, PDEControlEnv, make_env
from .task import TASK_SPECS, TaskSpec, make_task

# `pspe.envs.register_envs` used to register the flood family only, because it
# was the only family with a Gymnasium surface. It now registers all four. The
# per-module `register_envs` functions still exist and still register just their
# own family, which is what `tests/test_flood_env.py` calls.
register_envs = register_all

__all__ = [
    "GaussianActuatorBasis",
    "EnvSpec",
    "SPECS",
    "BY_ID",
    "catalog",
    "describe",
    "register_all",
    "register_envs",
    "FloodControlEnv",
    "FloodOperateEnv",
    "FloodOperateTask",
    "EvacuateControlEnv",
    "FirebreakEnv",
    "make_polder_bank",
    "SUMP_M",
    "RIVER_BED_M",
    "BatchedFieldEnv",
    "PDEControlEnv",
    "make_env",
    "TaskSpec",
    "TASK_SPECS",
    "make_task",
]
