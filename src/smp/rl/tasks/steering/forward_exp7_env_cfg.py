"""Experiment 7 forward configs: exp6 group 7 command plus standstill rewards."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg

from smp.rl.rewards import task_smp_product
from smp.rl.tasks.steering import mdp
from smp.rl.tasks.steering.forward_exp6_group7_env_cfg import (
  g1_forward_exp6_group7_smp_env_cfg,
)

_STOP_STATIC_CSV = Path("datasets/csv/forward/stop_static.csv")
_ZERO_THRESHOLD = 0.2
_FEET_CONTACT_SENSOR = "exp7_feet_contact"
_STOP_STATIC_FALLBACK_JOINT_POS = (
  0.05018,
  0.13543,
  0.278049,
  0.256678,
  -0.092145,
  -0.100208,
  0.092932,
  -0.132247,
  -0.244191,
  0.273684,
  -0.195645,
  0.05755,
  0.040542,
  0.014096,
  0.100762,
  0.197007,
  -0.001769,
  0.134267,
  0.66546,
  1.181996,
  0.210237,
  -0.059371,
  0.250677,
  -0.045274,
  -0.099932,
  0.769786,
  -0.939145,
  0.253011,
  0.155823,
)


@dataclass(frozen=True)
class Exp7ForwardGroup:
  group: int
  reward_name: str
  param_name: str
  reward_func: Callable[..., torch.Tensor]
  weight: float
  params: dict
  run_name: str
  summary: str


def _load_stop_static_joint_pos(path: Path = _STOP_STATIC_CSV) -> tuple[float, ...]:
  if not path.exists():
    return _STOP_STATIC_FALLBACK_JOINT_POS
  data = np.loadtxt(path, delimiter=",", ndmin=2)
  if data.shape[1] < 36:
    raise ValueError(f"Expected at least 36 columns in '{path}', got {data.shape[1]}.")
  return tuple(float(x) for x in data[0, 7:36])


def _feet_contact_sensor_cfg() -> ContactSensorCfg:
  return ContactSensorCfg(
    name=_FEET_CONTACT_SENSOR,
    primary=ContactMatch(
      mode="geom",
      pattern=r"^(left|right)_foot[1-7]_collision$",
      entity="robot",
    ),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )


def _standstill_params(
  *,
  command_name: str = "steering",
  default_joint_pos: tuple[float, ...] | None = None,
  joint_pos_scale: float | None = None,
  joint_vel_scale: float | None = None,
  foot_vel_scale: float | None = None,
  force_threshold: float | None = None,
) -> dict:
  params = {
    "command_name": command_name,
    "zero_threshold": _ZERO_THRESHOLD,
  }
  if default_joint_pos is not None:
    params["default_joint_pos"] = default_joint_pos
  if joint_pos_scale is not None:
    params["joint_pos_scale"] = joint_pos_scale
  if joint_vel_scale is not None:
    params["joint_vel_scale"] = joint_vel_scale
  if foot_vel_scale is not None:
    params["foot_vel_scale"] = foot_vel_scale
  if force_threshold is not None:
    params["force_threshold"] = force_threshold
    params["sensor_name"] = _FEET_CONTACT_SENSOR
  return params


_DEFAULT_JOINT_POS = _load_stop_static_joint_pos()

EXP7_FORWARD_GROUPS: tuple[Exp7ForwardGroup, ...] = (
  Exp7ForwardGroup(
    group=1,
    reward_name="stand_still_exp",
    param_name="weak",
    reward_func=mdp.stand_still_exp,
    weight=2.0,
    params=_standstill_params(
      default_joint_pos=_DEFAULT_JOINT_POS,
      joint_pos_scale=1.0,
    ),
    run_name="group1_stand_still_exp_w2",
    summary="stand_still_exp, w=2.0, zero_threshold=0.2, pose=stop_static[0]",
  ),
  Exp7ForwardGroup(
    group=2,
    reward_name="stand_still_exp",
    param_name="mid",
    reward_func=mdp.stand_still_exp,
    weight=5.0,
    params=_standstill_params(
      default_joint_pos=_DEFAULT_JOINT_POS,
      joint_pos_scale=1.0,
    ),
    run_name="group2_stand_still_exp_w5",
    summary="stand_still_exp, w=5.0, zero_threshold=0.2, pose=stop_static[0]",
  ),
  Exp7ForwardGroup(
    group=3,
    reward_name="stand_still_exp",
    param_name="strong",
    reward_func=mdp.stand_still_exp,
    weight=7.0,
    params=_standstill_params(
      default_joint_pos=_DEFAULT_JOINT_POS,
      joint_pos_scale=1.0,
    ),
    run_name="group3_stand_still_exp_w7",
    summary="stand_still_exp, w=7.0, zero_threshold=0.2, pose=stop_static[0]",
  ),
  Exp7ForwardGroup(
    group=4,
    reward_name="stand_still_vel",
    param_name="weak",
    reward_func=mdp.stand_still_vel,
    weight=-0.02,
    params=_standstill_params(joint_vel_scale=1.0),
    run_name="group4_stand_still_vel_wneg002",
    summary="stand_still_vel, w=-0.02, zero_threshold=0.2",
  ),
  Exp7ForwardGroup(
    group=5,
    reward_name="stand_still_vel",
    param_name="mid",
    reward_func=mdp.stand_still_vel,
    weight=-0.05,
    params=_standstill_params(joint_vel_scale=1.0),
    run_name="group5_stand_still_vel_wneg005",
    summary="stand_still_vel, w=-0.05, zero_threshold=0.2",
  ),
  Exp7ForwardGroup(
    group=6,
    reward_name="stand_still_vel",
    param_name="strong",
    reward_func=mdp.stand_still_vel,
    weight=-0.10,
    params=_standstill_params(joint_vel_scale=1.0),
    run_name="group6_stand_still_vel_wneg010",
    summary="stand_still_vel, w=-0.10, zero_threshold=0.2",
  ),
  Exp7ForwardGroup(
    group=7,
    reward_name="stand_still_feet_motion_penalty",
    param_name="weak",
    reward_func=mdp.stand_still_feet_motion_penalty,
    weight=-0.2,
    params=_standstill_params(foot_vel_scale=1.0),
    run_name="group7_stand_still_feet_wneg02",
    summary="stand_still_feet_motion_penalty, w=-0.2, zero_threshold=0.2",
  ),
  Exp7ForwardGroup(
    group=8,
    reward_name="stand_still_feet_motion_penalty",
    param_name="mid",
    reward_func=mdp.stand_still_feet_motion_penalty,
    weight=-0.5,
    params=_standstill_params(foot_vel_scale=1.0),
    run_name="group8_stand_still_feet_wneg05",
    summary="stand_still_feet_motion_penalty, w=-0.5, zero_threshold=0.2",
  ),
  Exp7ForwardGroup(
    group=9,
    reward_name="stand_still_feet_motion_penalty",
    param_name="strong",
    reward_func=mdp.stand_still_feet_motion_penalty,
    weight=-1.0,
    params=_standstill_params(foot_vel_scale=1.0),
    run_name="group9_stand_still_feet_wneg10",
    summary="stand_still_feet_motion_penalty, w=-1.0, zero_threshold=0.2",
  ),
  Exp7ForwardGroup(
    group=10,
    reward_name="stand_still_double_support",
    param_name="weak",
    reward_func=mdp.stand_still_double_support,
    weight=0.25,
    params=_standstill_params(force_threshold=2.0),
    run_name="group10_stand_still_double_support_w025",
    summary="stand_still_double_support, w=0.25, force_z>2.0, zero_threshold=0.2",
  ),
  Exp7ForwardGroup(
    group=11,
    reward_name="stand_still_double_support",
    param_name="mid",
    reward_func=mdp.stand_still_double_support,
    weight=0.5,
    params=_standstill_params(force_threshold=2.0),
    run_name="group11_stand_still_double_support_w05",
    summary="stand_still_double_support, w=0.5, force_z>2.0, zero_threshold=0.2",
  ),
  Exp7ForwardGroup(
    group=12,
    reward_name="stand_still_double_support",
    param_name="strong",
    reward_func=mdp.stand_still_double_support,
    weight=1.0,
    params=_standstill_params(force_threshold=2.0),
    run_name="group12_stand_still_double_support_w10",
    summary="stand_still_double_support, w=1.0, force_z>2.0, zero_threshold=0.2",
  ),
)


def _build_exp7_cfg(spec: Exp7ForwardGroup, play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = g1_forward_exp6_group7_smp_env_cfg(play=play)

  task_terms = (
    (
      mdp.steering_target_velocity,
      1.0,
      {"command_name": "steering", "vel_err_scale": 0.5},
    ),
    (spec.reward_func, spec.weight, spec.params),
  )
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_product,
    weight=1.0,
    params={"task_terms": task_terms},
  )

  if spec.reward_name == "stand_still_double_support":
    cfg.scene.sensors = (*cfg.scene.sensors, _feet_contact_sensor_cfg())

  return cfg


def _make_builder(spec: Exp7ForwardGroup) -> Callable[[bool], ManagerBasedRlEnvCfg]:
  def _builder(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _build_exp7_cfg(spec, play=play)

  _builder.__name__ = f"g1_forward_exp7_group{spec.group}_smp_env_cfg"
  return _builder


EXP7_FORWARD_BUILDERS = tuple(
  (spec.group, _make_builder(spec), spec) for spec in EXP7_FORWARD_GROUPS
)
