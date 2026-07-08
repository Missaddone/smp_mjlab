"""Experiment 6 group 12 forward config: zero + walking/running bins."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from smp.rl.tasks.steering.forward_env_cfg import g1_forward_smp_env_cfg


def g1_forward_exp6_group12_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = g1_forward_smp_env_cfg(play=play)
  command = cfg.commands["steering"]
  command.tar_speed_min = 1.5
  command.tar_speed_max = 5.0
  command.zero_speed_prob = 0.5
  command.low_speed_prob = 0.25
  command.low_speed_min = 0.0
  command.low_speed_max = 1.5
  return cfg
