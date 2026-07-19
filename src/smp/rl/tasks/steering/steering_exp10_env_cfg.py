"""Experiment 10 steering configs: original steering task with stop commands."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg

from smp.rl.rewards import task_smp_product
from smp.rl.tasks.steering import mdp
from smp.rl.tasks.steering.steering_env_cfg import g1_steering_smp_env_cfg

_PRIOR_CKPT = "datasets/pretrain_ckpt/exp10_loco_stop_static.pt"
_ZERO_SPEED_PROB = 0.3
_DEAD_ZONE_SPEED = 0.5
_ZERO_THRESHOLD = 0.2
_VEL_ERR_SCALE = 1.0
_ROOT_VEL_EXP_SCALE = 2.0
_JOINT_VEL_EXP_SCALE = 0.02


@dataclass(frozen=True)
class Exp10SteeringCommandSpec:
  key: str
  name: str
  tar_speed_max: float
  dead_zone_speed: float = 0.0


@dataclass(frozen=True)
class Exp10SteeringRewardSpec:
  key: str
  name: str
  summary: str
  task_terms: tuple


@dataclass(frozen=True)
class Exp10SteeringGroupSpec:
  group: int
  command: Exp10SteeringCommandSpec
  reward: Exp10SteeringRewardSpec
  run_name: str

  @property
  def reward_zero_threshold(self) -> float:
    return self.command.dead_zone_speed if self.command.dead_zone_speed > 0.0 else _ZERO_THRESHOLD


EXP10_STEERING_COMMANDS: tuple[Exp10SteeringCommandSpec, ...] = (
  Exp10SteeringCommandSpec("c1", "zero03_speed0_2", tar_speed_max=2.0),
  Exp10SteeringCommandSpec("c2", "zero03_speed0_4", tar_speed_max=4.0),
  Exp10SteeringCommandSpec(
    "c3",
    "zero03_speed0_2_dz05",
    tar_speed_max=2.0,
    dead_zone_speed=_DEAD_ZONE_SPEED,
  ),
  Exp10SteeringCommandSpec(
    "c4",
    "zero03_speed0_4_dz05",
    tar_speed_max=4.0,
    dead_zone_speed=_DEAD_ZONE_SPEED,
  ),
)

EXP10_STEERING_REWARDS: tuple[Exp10SteeringRewardSpec, ...] = (
  Exp10SteeringRewardSpec(
    key="c",
    name="stop_sum_switch",
    summary=(
      "moving uses original steering task; stopping uses "
      "0.6*r_root_stop + 0.4*r_joint_vel"
    ),
    task_terms=(
      (
        mdp.steering_stop_switch_sum_task,
        1.0,
        {
          "command_name": "steering",
          "vel_err_scale": _VEL_ERR_SCALE,
          "zero_threshold": _ZERO_THRESHOLD,
          "root_vel_exp_scale": _ROOT_VEL_EXP_SCALE,
          "joint_vel_exp_scale": _JOINT_VEL_EXP_SCALE,
          "moving_vel_weight": 0.5,
          "moving_face_weight": 0.5,
          "root_stop_weight": 0.6,
          "joint_vel_weight": 0.4,
        },
      ),
    ),
  ),
  Exp10SteeringRewardSpec(
    key="d",
    name="stop_product_switch",
    summary="moving uses original steering task; stopping uses r_root_stop*r_joint_vel",
    task_terms=(
      (
        mdp.steering_stop_switch_product_task,
        1.0,
        {
          "command_name": "steering",
          "vel_err_scale": _VEL_ERR_SCALE,
          "zero_threshold": _ZERO_THRESHOLD,
          "root_vel_exp_scale": _ROOT_VEL_EXP_SCALE,
          "joint_vel_exp_scale": _JOINT_VEL_EXP_SCALE,
          "moving_vel_weight": 0.5,
          "moving_face_weight": 0.5,
        },
      ),
    ),
  ),
  Exp10SteeringRewardSpec(
    key="e",
    name="stop_product_mix_switch",
    summary=(
      "moving uses original steering task; stopping uses "
      "0.6*r_root_stop*r_joint_vel + 0.2*r_root_stop + 0.2*r_joint_vel"
    ),
    task_terms=(
      (
        mdp.steering_stop_switch_mix_product_task,
        1.0,
        {
          "command_name": "steering",
          "vel_err_scale": _VEL_ERR_SCALE,
          "zero_threshold": _ZERO_THRESHOLD,
          "root_vel_exp_scale": _ROOT_VEL_EXP_SCALE,
          "joint_vel_exp_scale": _JOINT_VEL_EXP_SCALE,
          "moving_vel_weight": 0.5,
          "moving_face_weight": 0.5,
          "product_weight": 0.6,
          "root_stop_weight": 0.2,
          "joint_vel_weight": 0.2,
        },
      ),
    ),
  ),
)

EXP10_STEERING_GROUP_SPECS: tuple[Exp10SteeringGroupSpec, ...] = tuple(
  Exp10SteeringGroupSpec(
    group=command_idx * len(EXP10_STEERING_REWARDS) + reward_idx + 1,
    command=command,
    reward=reward,
    run_name=(
      f"group{command_idx * len(EXP10_STEERING_REWARDS) + reward_idx + 1:02d}_"
      f"steering_{command.key}_{reward.key}_{command.name}_{reward.name}"
    ),
  )
  for command_idx, command in enumerate(EXP10_STEERING_COMMANDS)
  for reward_idx, reward in enumerate(EXP10_STEERING_REWARDS)
)


def _build_exp10_steering_cfg(
  spec: Exp10SteeringGroupSpec, play: bool = False
) -> ManagerBasedRlEnvCfg:
  cfg = g1_steering_smp_env_cfg(play=play)
  command = cfg.commands["steering"]
  command.tar_speed_min = 0.0
  command.tar_speed_max = spec.command.tar_speed_max
  command.zero_speed_prob = _ZERO_SPEED_PROB
  command.dead_zone_speed = spec.command.dead_zone_speed
  command.zero_command_when_zero_speed = True
  task_terms = tuple(
    (func, weight, {**params, "zero_threshold": spec.reward_zero_threshold})
    for func, weight, params in spec.reward.task_terms
  )
  cfg.events["init_smp_state"].params["ckpt_path"] = _PRIOR_CKPT
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_product,
    weight=1.0,
    params={"task_terms": task_terms},
  )
  return cfg


def _make_builder(
  spec: Exp10SteeringGroupSpec,
) -> Callable[[bool], ManagerBasedRlEnvCfg]:
  def _builder(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _build_exp10_steering_cfg(spec, play=play)

  _builder.__name__ = f"g1_steering_exp10_group{spec.group}_smp_env_cfg"
  return _builder


EXP10_STEERING_BUILDERS = tuple(
  (spec.group, _make_builder(spec), spec) for spec in EXP10_STEERING_GROUP_SPECS
)
