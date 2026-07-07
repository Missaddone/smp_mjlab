"""Experiment 6 group 6 forward config: uniform speed with stop dead zone."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from smp.rl.tasks.steering.forward_env_cfg import g1_forward_smp_env_cfg


def g1_forward_exp6_group6_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = g1_forward_smp_env_cfg(play=play)
  command = cfg.commands["steering"]
  command.tar_speed_min = 0.0
  command.tar_speed_max = 5.0
  command.dead_zone_speed = 1.0
  return cfg
