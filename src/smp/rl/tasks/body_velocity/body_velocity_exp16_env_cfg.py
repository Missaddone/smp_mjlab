"""Experiment 16 body-velocity policies trained from the motebu/static prior."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg

from smp.rl.tasks.body_velocity.body_velocity_exp13_env_cfg import (
  EXP13_BODY_VELOCITY_GROUP_SPECS,
  _build_exp13_body_velocity_cfg,
)

_PRIOR_CKPT = "datasets/pretrain_ckpt/exp16_motebu_stop_static.pt"
_COMMAND_LIN_VEL_ABS_MAX = 1.5
_COMMAND_YAW_RATE_ABS_MAX = 2.0
_ZERO_COMMAND_PROB = 0.3


@dataclass(frozen=True)
class Exp16BodyVelocityGroupSpec:
  group: int
  exp13_reward_group: int
  parent_group: None
  run_name: str
  summary: str


EXP16_BODY_VELOCITY_GROUP_SPECS: tuple[Exp16BodyVelocityGroupSpec, ...] = tuple(
  Exp16BodyVelocityGroupSpec(
    group=group,
    exp13_reward_group=exp13_reward_group,
    parent_group=None,
    run_name=f"group{group:02d}_motebu_stop_static_exp13_g{exp13_reward_group}_reward",
    summary=(
      "Train from scratch with the motebu/stop-static mirrored prior and "
      f"the Exp13 G{exp13_reward_group} moving-reward mix"
    ),
  )
  for group, exp13_reward_group in enumerate((4, 5, 6), start=1)
)


def _exp13_spec(group: int):
  return next(spec for spec in EXP13_BODY_VELOCITY_GROUP_SPECS if spec.group == group)


def _build_exp16_body_velocity_cfg(
  spec: Exp16BodyVelocityGroupSpec, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Reuse Exp13 G4--G6 reward recipes with an Exp16 prior and commands."""
  cfg = _build_exp13_body_velocity_cfg(
    _exp13_spec(spec.exp13_reward_group), play=play
  )
  command = cfg.commands["body_velocity"]
  command.lin_vel_x_min = -_COMMAND_LIN_VEL_ABS_MAX
  command.lin_vel_x_max = _COMMAND_LIN_VEL_ABS_MAX
  command.lin_vel_y_min = -_COMMAND_LIN_VEL_ABS_MAX
  command.lin_vel_y_max = _COMMAND_LIN_VEL_ABS_MAX
  command.yaw_rate_min = -_COMMAND_YAW_RATE_ABS_MAX
  command.yaw_rate_max = _COMMAND_YAW_RATE_ABS_MAX
  command.zero_command_prob = _ZERO_COMMAND_PROB
  cfg.events["init_smp_state"].params["ckpt_path"] = _PRIOR_CKPT
  return cfg


def _make_builder(
  spec: Exp16BodyVelocityGroupSpec,
) -> Callable[[bool], ManagerBasedRlEnvCfg]:
  def _builder(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _build_exp16_body_velocity_cfg(spec, play=play)

  _builder.__name__ = f"g1_body_velocity_exp16_group{spec.group}_smp_env_cfg"
  return _builder


EXP16_BODY_VELOCITY_BUILDERS = tuple(
  (spec.group, _make_builder(spec), spec)
  for spec in EXP16_BODY_VELOCITY_GROUP_SPECS
)
