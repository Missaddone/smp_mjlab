"""Experiment 6 group 8 forward config: zero-speed mixture p=0.4."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from smp.rl.tasks.steering.forward_env_cfg import g1_forward_smp_env_cfg


def g1_forward_exp6_group8_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = g1_forward_smp_env_cfg(play=play)
  command = cfg.commands["steering"]
  command.tar_speed_min = 0.0
  command.tar_speed_max = 5.0
  command.zero_speed_prob = 0.4
  return cfg
