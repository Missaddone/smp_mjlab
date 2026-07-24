"""Experiment 12 body-velocity configs for theme/style priors."""

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

_BASE_EXP10_GROUP = 14
_FEET_CONTACT_SENSOR = "exp12_feet_ground_contact"
_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
_FOOT_TILT_WEIGHTS = (-0.05, -0.1, -0.2, -0.3)
_MOVING_PRODUCT_WEIGHT = 0.6
_MOVING_LINEAR_WEIGHT = 0.2
_MOVING_YAW_WEIGHT = 0.2
_PERSISTENT_SINGLE_SUPPORT_WEIGHT = -0.2
_DOUBLE_AIR_WEIGHT = -0.3
_COMMAND_ACTIVE_THRESHOLD = 0.2
_MAX_SINGLE_SUPPORT_TIME = 0.45
_MAX_SINGLE_SUPPORT_EXCESS_TIME = 0.8


@dataclass(frozen=True)
class Exp12BodyVelocityGroupSpec:
  group: int
  style: str
  prior_ckpt: str
  base_exp10_group: int
  foot_tilt_weight: float | None
  add_theme_regularizers: bool
  run_name: str
  summary: str


def _exp10_group_spec(group: int):
  return next(spec for spec in EXP10_BODY_VELOCITY_GROUP_SPECS if spec.group == group)


def _feet_contact_sensor_cfg() -> ContactSensorCfg:
  return ContactSensorCfg(
    name=_FEET_CONTACT_SENSOR,
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )


def _append_feet_contact_sensor(cfg: ManagerBasedRlEnvCfg) -> None:
  if any(sensor.name == _FEET_CONTACT_SENSOR for sensor in cfg.scene.sensors):
    return
  cfg.scene.sensors = (*cfg.scene.sensors, _feet_contact_sensor_cfg())


def _weight_name(weight: float) -> str:
  return f"w{abs(int(round(weight * 100))):03d}"


_STYLE_PRIORS: tuple[tuple[str, str], ...] = (
  ("male", "datasets/pretrain_ckpt/exp12_theme_male.pt"),
  ("female", "datasets/pretrain_ckpt/exp12_theme_female.pt"),
  ("children", "datasets/pretrain_ckpt/exp12_theme_children.pt"),
)


_EXP12_THEME_REGULARIZER_SPECS: tuple[Exp12BodyVelocityGroupSpec, ...] = tuple(
  Exp12BodyVelocityGroupSpec(
    group=group,
    style=style,
    prior_ckpt=prior_ckpt,
    base_exp10_group=_BASE_EXP10_GROUP,
    foot_tilt_weight=weight,
    add_theme_regularizers=True,
    run_name=(
      f"group{group:02d}_theme_{style}_exp10_g14_"
      f"moving_mix060_gait_support_foot_tilt_{_weight_name(weight)}"
    ),
    summary=(
      f"Exp10 group14 config, {style} theme prior, "
      "moving reward 0.6*r_l*r_y + 0.2*r_l + 0.2*r_y, "
      f"{weight}*support_foot_tilt_penalty, -0.2*single_support, -0.3*double_air"
    ),
  )
  for group, (style, prior_ckpt, weight) in enumerate(
    (
      (style, prior_ckpt, weight)
      for style, prior_ckpt in _STYLE_PRIORS
      for weight in _FOOT_TILT_WEIGHTS
    ),
    start=1,
  )
)


_EXP12_PRIOR_ONLY_SPECS: tuple[Exp12BodyVelocityGroupSpec, ...] = tuple(
  Exp12BodyVelocityGroupSpec(
    group=group,
    style=style,
    prior_ckpt=prior_ckpt,
    base_exp10_group=base_group,
    foot_tilt_weight=None,
    add_theme_regularizers=False,
    run_name=f"group{group:02d}_theme_{style}_exp10_g{base_group}_prior_only",
    summary=(
      f"Exp10 group{base_group} config, {style} theme prior only; "
      "command, moving reward, stop reward, observations, and extra rewards "
      f"inherit Exp10 group{base_group}"
    ),
  )
  for group, (base_group, style, prior_ckpt) in enumerate(
    (
      (base_group, style, prior_ckpt)
      for base_group in (14, 15)
      for style, prior_ckpt in _STYLE_PRIORS
    ),
    start=13,
  )
)


EXP12_BODY_VELOCITY_GROUP_SPECS: tuple[Exp12BodyVelocityGroupSpec, ...] = (
  *_EXP12_THEME_REGULARIZER_SPECS,
  *_EXP12_PRIOR_ONLY_SPECS,
)


def _build_exp12_body_velocity_cfg(
  spec: Exp12BodyVelocityGroupSpec, play: bool = False
) -> ManagerBasedRlEnvCfg:
  cfg = _build_exp10_body_velocity_cfg(_exp10_group_spec(spec.base_exp10_group), play=play)
  cfg.events["init_smp_state"].params["ckpt_path"] = spec.prior_ckpt
  if not spec.add_theme_regularizers:
    return cfg

  set_body_velocity_moving_product_mix(
    cfg,
    product_weight=_MOVING_PRODUCT_WEIGHT,
    linear_weight=_MOVING_LINEAR_WEIGHT,
    yaw_weight=_MOVING_YAW_WEIGHT,
  )
  _append_feet_contact_sensor(cfg)
  cfg.rewards["support_foot_tilt"] = RewardTermCfg(
    func=mdp.support_foot_tilt_penalty,
    weight=spec.foot_tilt_weight or 0.0,
    params={
      "sensor_name": _FEET_CONTACT_SENSOR,
      "contact_threshold": 1.0,
      "asset_cfg": SceneEntityCfg("robot", body_names=_FOOT_BODY_NAMES),
    },
  )
  cfg.rewards["persistent_single_support"] = RewardTermCfg(
    func=mdp.persistent_single_support_penalty,
    weight=_PERSISTENT_SINGLE_SUPPORT_WEIGHT,
    params={
      "command_name": "body_velocity",
      "sensor_name": _FEET_CONTACT_SENSOR,
      "max_single_support_time": _MAX_SINGLE_SUPPORT_TIME,
      "max_excess_time": _MAX_SINGLE_SUPPORT_EXCESS_TIME,
      "command_threshold": _COMMAND_ACTIVE_THRESHOLD,
    },
  )
  cfg.rewards["double_air"] = RewardTermCfg(
    func=mdp.double_air_penalty,
    weight=_DOUBLE_AIR_WEIGHT,
    params={
      "command_name": "body_velocity",
      "sensor_name": _FEET_CONTACT_SENSOR,
      "command_threshold": _COMMAND_ACTIVE_THRESHOLD,
    },
  )
  return cfg


def _make_builder(
  spec: Exp12BodyVelocityGroupSpec,
) -> Callable[[bool], ManagerBasedRlEnvCfg]:
  def _builder(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _build_exp12_body_velocity_cfg(spec, play=play)

  _builder.__name__ = f"g1_body_velocity_exp12_group{spec.group}_smp_env_cfg"
  return _builder


EXP12_BODY_VELOCITY_BUILDERS = tuple(
  (spec.group, _make_builder(spec), spec)
  for spec in EXP12_BODY_VELOCITY_GROUP_SPECS
)
