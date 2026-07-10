"""Experiment 6 group 13 forward config: group 8 command with new prior."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from smp.rl.tasks.steering.forward_exp6_group8_env_cfg import (
  g1_forward_exp6_group8_smp_env_cfg,
)


def g1_forward_exp6_group13_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return g1_forward_exp6_group8_smp_env_cfg(play=play)
