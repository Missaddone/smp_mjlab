"""Experiment 10 body-velocity configs: body-frame commands with stop branches."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg

from smp.rl.rewards import body_velocity_task_smp_product
from smp.rl.tasks.body_velocity import mdp
from smp.rl.tasks.body_velocity.body_velocity_env_cfg import (
  g1_body_velocity_smp_env_cfg,
)

_PRIOR_CKPT = "datasets/pretrain_ckpt/exp10_loco_stop_static.pt"
_ZERO_COMMAND_PROB = 0.3
_DEAD_ZONE_SPEED = 0.5
_COMMAND_ZERO_THRESHOLD = 0.2
_LIN_VEL_ERR_SCALE = 2.0
_YAW_RATE_ERR_SCALE = 1.0
_ROOT_VEL_EXP_SCALE = 2.0
_JOINT_VEL_EXP_SCALE = 0.02


@dataclass(frozen=True)
class Exp10BodyVelocityCommandSpec:
  key: str
  name: str
  lin_vel_abs_max: float
  yaw_rate_abs_max: float
  dead_zone_speed: float = 0.0


@dataclass(frozen=True)
class Exp10BodyVelocityRewardSpec:
  key: str
  name: str
  summary: str
  task_terms: tuple


@dataclass(frozen=True)
class Exp10BodyVelocityGroupSpec:
  group: int
  command: Exp10BodyVelocityCommandSpec
  reward: Exp10BodyVelocityRewardSpec
  run_name: str

  @property
  def reward_command_zero_threshold(self) -> float:
    if self.command.dead_zone_speed > 0.0:
      return self.command.dead_zone_speed
    return _COMMAND_ZERO_THRESHOLD


EXP10_BODY_VELOCITY_COMMANDS: tuple[Exp10BodyVelocityCommandSpec, ...] = (
  Exp10BodyVelocityCommandSpec(
    "c1",
    "zero03_xy2_yaw1",
    lin_vel_abs_max=2.0,
    yaw_rate_abs_max=1.0,
  ),
  Exp10BodyVelocityCommandSpec(
    "c2",
    "zero03_xy4_yaw2",
    lin_vel_abs_max=4.0,
    yaw_rate_abs_max=2.0,
  ),
  Exp10BodyVelocityCommandSpec(
    "c3",
    "zero03_xy2_yaw1_dz05",
    lin_vel_abs_max=2.0,
    yaw_rate_abs_max=1.0,
    dead_zone_speed=_DEAD_ZONE_SPEED,
  ),
  Exp10BodyVelocityCommandSpec(
    "c4",
    "zero03_xy4_yaw2_dz05",
    lin_vel_abs_max=4.0,
    yaw_rate_abs_max=2.0,
    dead_zone_speed=_DEAD_ZONE_SPEED,
  ),
)

EXP10_BODY_VELOCITY_REWARDS: tuple[Exp10BodyVelocityRewardSpec, ...] = (
  Exp10BodyVelocityRewardSpec(
    key="c",
    name="stop_sum_switch",
    summary=(
      "moving uses original body-velocity task; stopping uses "
      "0.6*r_root_stop + 0.4*r_joint_vel"
    ),
    task_terms=(
      (
        mdp.body_velocity_stop_switch_sum_task,
        1.0,
        {
          "command_name": "body_velocity",
          "lin_vel_err_scale": _LIN_VEL_ERR_SCALE,
          "yaw_rate_err_scale": _YAW_RATE_ERR_SCALE,
          "command_zero_threshold": _COMMAND_ZERO_THRESHOLD,
          "root_vel_exp_scale": _ROOT_VEL_EXP_SCALE,
          "joint_vel_exp_scale": _JOINT_VEL_EXP_SCALE,
          "moving_linear_weight": 0.75,
          "moving_yaw_weight": 0.25,
          "root_stop_weight": 0.6,
          "joint_vel_weight": 0.4,
        },
      ),
    ),
  ),
  Exp10BodyVelocityRewardSpec(
    key="d",
    name="stop_product_switch",
    summary=(
      "moving uses original body-velocity task; stopping uses "
      "r_root_stop*r_joint_vel"
    ),
    task_terms=(
      (
        mdp.body_velocity_stop_switch_product_task,
        1.0,
        {
          "command_name": "body_velocity",
          "lin_vel_err_scale": _LIN_VEL_ERR_SCALE,
          "yaw_rate_err_scale": _YAW_RATE_ERR_SCALE,
          "command_zero_threshold": _COMMAND_ZERO_THRESHOLD,
          "root_vel_exp_scale": _ROOT_VEL_EXP_SCALE,
          "joint_vel_exp_scale": _JOINT_VEL_EXP_SCALE,
          "moving_linear_weight": 0.75,
          "moving_yaw_weight": 0.25,
        },
      ),
    ),
  ),
  Exp10BodyVelocityRewardSpec(
    key="e",
    name="stop_product_mix_switch",
    summary=(
      "moving uses original body-velocity task; stopping uses "
      "0.6*r_root_stop*r_joint_vel + 0.2*r_root_stop + 0.2*r_joint_vel"
    ),
    task_terms=(
      (
        mdp.body_velocity_stop_switch_mix_product_task,
        1.0,
        {
          "command_name": "body_velocity",
          "lin_vel_err_scale": _LIN_VEL_ERR_SCALE,
          "yaw_rate_err_scale": _YAW_RATE_ERR_SCALE,
          "command_zero_threshold": _COMMAND_ZERO_THRESHOLD,
          "root_vel_exp_scale": _ROOT_VEL_EXP_SCALE,
          "joint_vel_exp_scale": _JOINT_VEL_EXP_SCALE,
          "moving_linear_weight": 0.75,
          "moving_yaw_weight": 0.25,
          "product_weight": 0.6,
          "root_stop_weight": 0.2,
          "joint_vel_weight": 0.2,
        },
      ),
    ),
  ),
)

EXP10_BODY_VELOCITY_GROUP_SPECS: tuple[Exp10BodyVelocityGroupSpec, ...] = tuple(
  Exp10BodyVelocityGroupSpec(
    group=13 + command_idx * len(EXP10_BODY_VELOCITY_REWARDS) + reward_idx,
    command=command,
    reward=reward,
    run_name=(
      f"group{13 + command_idx * len(EXP10_BODY_VELOCITY_REWARDS) + reward_idx:02d}_"
      f"body_velocity_{command.key}_{reward.key}_{command.name}_{reward.name}"
    ),
  )
  for command_idx, command in enumerate(EXP10_BODY_VELOCITY_COMMANDS)
  for reward_idx, reward in enumerate(EXP10_BODY_VELOCITY_REWARDS)
)


def _build_exp10_body_velocity_cfg(
  spec: Exp10BodyVelocityGroupSpec, play: bool = False
) -> ManagerBasedRlEnvCfg:
  cfg = g1_body_velocity_smp_env_cfg(play=play)
  command = cfg.commands["body_velocity"]
  command.lin_vel_x_min = -spec.command.lin_vel_abs_max
  command.lin_vel_x_max = spec.command.lin_vel_abs_max
  command.lin_vel_y_min = -spec.command.lin_vel_abs_max
  command.lin_vel_y_max = spec.command.lin_vel_abs_max
  command.yaw_rate_min = -spec.command.yaw_rate_abs_max
  command.yaw_rate_max = spec.command.yaw_rate_abs_max
  command.zero_command_prob = _ZERO_COMMAND_PROB
  command.dead_zone_speed = spec.command.dead_zone_speed
  task_terms = tuple(
    (
      func,
      weight,
      {**params, "command_zero_threshold": spec.reward_command_zero_threshold},
    )
    for func, weight, params in spec.reward.task_terms
  )
  cfg.events["init_smp_state"].params["ckpt_path"] = _PRIOR_CKPT
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=body_velocity_task_smp_product,
    weight=1.0,
    params={
      "task_terms": task_terms,
      "fixed_timesteps": (8, 15, 22),
      "ws": 6.0,
    },
  )
  return cfg


def _make_builder(
  spec: Exp10BodyVelocityGroupSpec,
) -> Callable[[bool], ManagerBasedRlEnvCfg]:
  def _builder(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _build_exp10_body_velocity_cfg(spec, play=play)

  _builder.__name__ = f"g1_body_velocity_exp10_group{spec.group}_smp_env_cfg"
  return _builder


EXP10_BODY_VELOCITY_BUILDERS = tuple(
  (spec.group, _make_builder(spec), spec)
  for spec in EXP10_BODY_VELOCITY_GROUP_SPECS
)
