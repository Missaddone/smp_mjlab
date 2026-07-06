"""Experiment 4 group 4 body-velocity config."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from smp.rl.tasks.body_velocity.body_velocity_env_cfg import (
  g1_body_velocity_exp4_smp_env_cfg,
)


def g1_body_velocity_exp4_group4_smp_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return g1_body_velocity_exp4_smp_env_cfg(
    play=play,
    lin_vel_err_scale=1.0,
    yaw_rate_err_scale=0.5,
    product_weight=0.0,
    linear_weight=0.5,
    yaw_weight=0.5,
  )
