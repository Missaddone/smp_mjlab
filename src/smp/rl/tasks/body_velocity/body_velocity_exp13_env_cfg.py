"""Experiment 13 body-velocity moving-reward product-mix ablations."""

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

_FEET_CONTACT_SENSOR = "exp13_feet_ground_contact"
_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
_COMMAND_ACTIVE_THRESHOLD = 0.2
_MAX_SINGLE_SUPPORT_TIME = 0.45
_MAX_SINGLE_SUPPORT_EXCESS_TIME = 0.8


@dataclass(frozen=True)
class Exp13MovingRewardSpec:
  key: str
  product_weight: float
  linear_weight: float
  yaw_weight: float


@dataclass(frozen=True)
class Exp13BodyVelocityGroupSpec:
  group: int
  base_exp10_group: int
  reward: Exp13MovingRewardSpec
  foot_tilt_weight: float
  persistent_single_support_weight: float
  finetune_parent_group: int | None
  run_name: str
  summary: str


_MOVING_REWARD_SPECS: tuple[Exp13MovingRewardSpec, ...] = (
  Exp13MovingRewardSpec("mix050_025_025", 0.5, 0.25, 0.25),
  Exp13MovingRewardSpec("mix060_020_020", 0.6, 0.2, 0.2),
  Exp13MovingRewardSpec("mix070_015_015", 0.7, 0.15, 0.15),
)


def _exp10_spec(base_group: int):
  return next(spec for spec in EXP10_BODY_VELOCITY_GROUP_SPECS if spec.group == base_group)


_BASE_GROUP_SPECS: tuple[Exp13BodyVelocityGroupSpec, ...] = tuple(
  Exp13BodyVelocityGroupSpec(
    group=group,
    base_exp10_group=base_group,
    reward=reward,
    foot_tilt_weight=0.0,
    persistent_single_support_weight=0.0,
    finetune_parent_group=None,
    run_name=(
      f"group{group:02d}_from_exp10_g{base_group}_moving_{reward.key}"
    ),
    summary=(
      f"Exp10 group{base_group} unchanged except moving reward "
      f"{reward.product_weight}*r_l*r_y + {reward.linear_weight}*r_l + "
      f"{reward.yaw_weight}*r_y"
    ),
  )
  for group, (base_group, reward) in enumerate(
    (
      *((14, reward) for reward in _MOVING_REWARD_SPECS),
      *((15, reward) for reward in _MOVING_REWARD_SPECS),
    ),
    start=1,
  )
)


def _finetune_specs_for_parent(
  parent: Exp13BodyVelocityGroupSpec, *, start_group: int
) -> tuple[Exp13BodyVelocityGroupSpec, ...]:
  variants = (
    ("g20_style", -0.1, -0.4),
    ("foot_tilt_w005", -0.05, 0.0),
    ("foot_tilt_w020", -0.2, 0.0),
    ("foot_tilt_w040", -0.4, 0.0),
  )
  specs: list[Exp13BodyVelocityGroupSpec] = []
  for group, (suffix, foot_tilt_weight, persistent_weight) in enumerate(
    variants, start=start_group
  ):
    terms = [f"{foot_tilt_weight}*support_foot_tilt"]
    if persistent_weight != 0.0:
      terms.append(f"{persistent_weight}*persistent_single_support")
    specs.append(
      Exp13BodyVelocityGroupSpec(
        group=group,
        base_exp10_group=parent.base_exp10_group,
        reward=parent.reward,
        foot_tilt_weight=foot_tilt_weight,
        persistent_single_support_weight=persistent_weight,
        finetune_parent_group=parent.group,
        run_name=f"group{group:02d}_finetune_exp13_g{parent.group}_{suffix}",
        summary=(
          f"Fine-tune Exp13 group{parent.group}; preserve its command, prior, "
          f"stop branch, and moving reward, plus {', '.join(terms)}"
        ),
      )
    )
  return tuple(specs)


_FINETUNE_GROUP_SPECS = tuple(
  spec
  for start_group, parent_group in ((7, 4), (11, 5), (15, 6))
  for spec in _finetune_specs_for_parent(
    next(spec for spec in _BASE_GROUP_SPECS if spec.group == parent_group),
    start_group=start_group,
  )
)

EXP13_BODY_VELOCITY_GROUP_SPECS = _BASE_GROUP_SPECS + _FINETUNE_GROUP_SPECS


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


def _build_exp13_body_velocity_cfg(
  spec: Exp13BodyVelocityGroupSpec, play: bool = False
) -> ManagerBasedRlEnvCfg:
  cfg = _build_exp10_body_velocity_cfg(_exp10_spec(spec.base_exp10_group), play=play)
  set_body_velocity_moving_product_mix(
    cfg,
    product_weight=spec.reward.product_weight,
    linear_weight=spec.reward.linear_weight,
    yaw_weight=spec.reward.yaw_weight,
  )
  if spec.foot_tilt_weight != 0.0:
    _append_feet_contact_sensor(
      cfg,
      track_air_time=spec.persistent_single_support_weight != 0.0,
    )
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
  return cfg


def _make_builder(
  spec: Exp13BodyVelocityGroupSpec,
) -> Callable[[bool], ManagerBasedRlEnvCfg]:
  def _builder(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _build_exp13_body_velocity_cfg(spec, play=play)

  _builder.__name__ = f"g1_body_velocity_exp13_group{spec.group}_smp_env_cfg"
  return _builder


EXP13_BODY_VELOCITY_BUILDERS = tuple(
  (spec.group, _make_builder(spec), spec)
  for spec in EXP13_BODY_VELOCITY_GROUP_SPECS
)
