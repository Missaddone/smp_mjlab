"""Experiment 11 body-velocity foot-tilt fine-tuning configs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg

from smp.rl.tasks.body_velocity import mdp
from smp.rl.tasks.body_velocity.body_velocity_exp10_env_cfg import (
  EXP10_BODY_VELOCITY_GROUP_SPECS,
  _build_exp10_body_velocity_cfg,
  set_body_velocity_moving_product_mix,
)

_FEET_CONTACT_SENSOR = "exp11_feet_ground_contact"
_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
_FOOT_TILT_WEIGHTS = (-0.05, -0.1, -0.2, -0.3)
_GROUP9_PLUS_FOOT_TILT_WEIGHT = -0.1
_MOVING_PRODUCT_WEIGHT = 0.6
_MOVING_LINEAR_WEIGHT = 0.2
_MOVING_YAW_WEIGHT = 0.2
_COMMAND_ACTIVE_THRESHOLD = 0.2
_MAX_SINGLE_SUPPORT_TIME = 0.45
_MAX_SINGLE_SUPPORT_EXCESS_TIME = 0.8


@dataclass(frozen=True)
class Exp11BodyVelocityGroupSpec:
  group: int
  base_exp10_group: int
  foot_tilt_weight: float
  moving_product_weight: float | None
  moving_linear_weight: float | None
  moving_yaw_weight: float | None
  persistent_single_support_weight: float
  double_air_weight: float
  run_name: str
  summary: str


def _feet_contact_sensor_cfg(*, track_air_time: bool) -> ContactSensorCfg:
  fields = ("found", "force") if track_air_time else ("force",)
  return ContactSensorCfg(
    name=_FEET_CONTACT_SENSOR,
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=fields,
    reduce="netforce",
    num_slots=1,
    track_air_time=track_air_time,
  )


def _append_feet_contact_sensor(
  cfg: ManagerBasedRlEnvCfg, *, track_air_time: bool
) -> None:
  if any(sensor.name == _FEET_CONTACT_SENSOR for sensor in cfg.scene.sensors):
    return
  cfg.scene.sensors = (
    *cfg.scene.sensors,
    _feet_contact_sensor_cfg(track_air_time=track_air_time),
  )


def _exp10_spec(base_group: int):
  return next(spec for spec in EXP10_BODY_VELOCITY_GROUP_SPECS if spec.group == base_group)


def _weight_name(weight: float) -> str:
  return f"w{abs(int(round(weight * 100))):03d}"


_EXP11_BASE_GROUP_SPECS: tuple[Exp11BodyVelocityGroupSpec, ...] = tuple(
  Exp11BodyVelocityGroupSpec(
    group=group_idx,
    base_exp10_group=base_group,
    foot_tilt_weight=weight,
    moving_product_weight=None,
    moving_linear_weight=None,
    moving_yaw_weight=None,
    persistent_single_support_weight=0.0,
    double_air_weight=0.0,
    run_name=(
      f"group{group_idx:02d}_from_exp10_g{base_group}_"
      f"support_foot_tilt_{_weight_name(weight)}"
    ),
    summary=(
      f"Exp10 group{base_group} unchanged, plus "
      f"{weight}*support_foot_tilt_penalty"
    ),
  )
  for group_idx, (base_group, weight) in enumerate(
    (
      *((14, weight) for weight in _FOOT_TILT_WEIGHTS),
      *((15, weight) for weight in _FOOT_TILT_WEIGHTS),
    ),
    start=1,
  )
)


_GAIT_ONLY_WEIGHTS = (-0.1, -0.2, -0.4)
_DROP_ONLY_WEIGHTS = (-0.1, -0.3, -0.6)
_GAIT_DROP_COMBOS = (
  (-0.4, -0.1),
  (-0.2, -0.3),
  (-0.1, -0.6),
)


def _extra_reward_specs_for_base(
  base_group: int, *, start_group: int
) -> tuple[Exp11BodyVelocityGroupSpec, ...]:
  specs: list[Exp11BodyVelocityGroupSpec] = []
  group = start_group
  for gait_weight in _GAIT_ONLY_WEIGHTS:
    specs.append(
      _extra_reward_spec(
        group=group,
        base_group=base_group,
        suffix=f"gait_{_weight_name(gait_weight)}",
        persistent_single_support_weight=gait_weight,
        double_air_weight=0.0,
      )
    )
    group += 1
  for double_air_weight in _DROP_ONLY_WEIGHTS:
    specs.append(
      _extra_reward_spec(
        group=group,
        base_group=base_group,
        suffix=f"drop_{_weight_name(double_air_weight)}",
        persistent_single_support_weight=0.0,
        double_air_weight=double_air_weight,
      )
    )
    group += 1
  for gait_weight, double_air_weight in _GAIT_DROP_COMBOS:
    specs.append(
      _extra_reward_spec(
        group=group,
        base_group=base_group,
        suffix=(
          f"gait_{_weight_name(gait_weight)}_"
          f"drop_{_weight_name(double_air_weight)}"
        ),
        persistent_single_support_weight=gait_weight,
        double_air_weight=double_air_weight,
      )
    )
    group += 1
  return tuple(specs)


def _extra_reward_spec(
  *,
  group: int,
  base_group: int,
  suffix: str,
  persistent_single_support_weight: float,
  double_air_weight: float,
) -> Exp11BodyVelocityGroupSpec:
  return Exp11BodyVelocityGroupSpec(
    group=group,
    base_exp10_group=base_group,
    foot_tilt_weight=_GROUP9_PLUS_FOOT_TILT_WEIGHT,
    moving_product_weight=_MOVING_PRODUCT_WEIGHT,
    moving_linear_weight=_MOVING_LINEAR_WEIGHT,
    moving_yaw_weight=_MOVING_YAW_WEIGHT,
    persistent_single_support_weight=persistent_single_support_weight,
    double_air_weight=double_air_weight,
    run_name=f"group{group:02d}_from_exp10_g{base_group}_moving_mix060_{suffix}",
    summary=(
      f"Exp10 group{base_group}, moving reward 0.6*r_l*r_y + 0.2*r_l + 0.2*r_y, "
      f"{_GROUP9_PLUS_FOOT_TILT_WEIGHT}*support_foot_tilt, "
      f"{persistent_single_support_weight}*persistent_single_support, "
      f"{double_air_weight}*double_air"
    ),
  )


EXP11_BODY_VELOCITY_GROUP_SPECS: tuple[Exp11BodyVelocityGroupSpec, ...] = (
  *_EXP11_BASE_GROUP_SPECS,
  *_extra_reward_specs_for_base(14, start_group=9),
  *_extra_reward_specs_for_base(15, start_group=18),
)


def _build_exp11_body_velocity_cfg(
  spec: Exp11BodyVelocityGroupSpec, play: bool = False
) -> ManagerBasedRlEnvCfg:
  cfg = _build_exp10_body_velocity_cfg(_exp10_spec(spec.base_exp10_group), play=play)
  if spec.moving_product_weight is not None:
    assert spec.moving_linear_weight is not None
    assert spec.moving_yaw_weight is not None
    set_body_velocity_moving_product_mix(
      cfg,
      product_weight=spec.moving_product_weight,
      linear_weight=spec.moving_linear_weight,
      yaw_weight=spec.moving_yaw_weight,
    )
  needs_air_time = (
    spec.persistent_single_support_weight != 0.0 or spec.double_air_weight != 0.0
  )
  _append_feet_contact_sensor(cfg, track_air_time=needs_air_time)
  cfg.rewards["support_foot_tilt"] = RewardTermCfg(
    func=mdp.support_foot_tilt_penalty,
    weight=spec.foot_tilt_weight,
    params={
      "sensor_name": _FEET_CONTACT_SENSOR,
      "contact_threshold": 1.0,
      "asset_cfg": SceneEntityCfg("robot", body_names=_FOOT_BODY_NAMES),
    },
  )
  if spec.persistent_single_support_weight != 0.0:
    cfg.rewards["persistent_single_support"] = RewardTermCfg(
      func=mdp.persistent_single_support_penalty,
      weight=spec.persistent_single_support_weight,
      params={
        "command_name": "body_velocity",
        "sensor_name": _FEET_CONTACT_SENSOR,
        "max_single_support_time": _MAX_SINGLE_SUPPORT_TIME,
        "max_excess_time": _MAX_SINGLE_SUPPORT_EXCESS_TIME,
        "command_threshold": _COMMAND_ACTIVE_THRESHOLD,
      },
    )
  if spec.double_air_weight != 0.0:
    cfg.rewards["double_air"] = RewardTermCfg(
      func=mdp.double_air_penalty,
      weight=spec.double_air_weight,
      params={
        "command_name": "body_velocity",
        "sensor_name": _FEET_CONTACT_SENSOR,
        "command_threshold": _COMMAND_ACTIVE_THRESHOLD,
      },
    )
  return cfg


def _make_builder(
  spec: Exp11BodyVelocityGroupSpec,
) -> Callable[[bool], ManagerBasedRlEnvCfg]:
  def _builder(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _build_exp11_body_velocity_cfg(spec, play=play)

  _builder.__name__ = f"g1_body_velocity_exp11_group{spec.group}_smp_env_cfg"
  return _builder


EXP11_BODY_VELOCITY_BUILDERS = tuple(
  (spec.group, _make_builder(spec), spec)
  for spec in EXP11_BODY_VELOCITY_GROUP_SPECS
)
