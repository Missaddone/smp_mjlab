"""Experiment 6 group 14 forward config: group 11 command with new prior."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from smp.rl.tasks.steering.forward_exp6_group11_env_cfg import (
  g1_forward_exp6_group11_smp_env_cfg,
)


def g1_forward_exp6_group14_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return g1_forward_exp6_group11_smp_env_cfg(play=play)
