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
)

_FEET_CONTACT_SENSOR = "exp11_feet_ground_contact"
_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
_FOOT_TILT_WEIGHTS = (-0.05, -0.1, -0.2, -0.3)


@dataclass(frozen=True)
class Exp11BodyVelocityGroupSpec:
  group: int
  base_exp10_group: int
  foot_tilt_weight: float
  run_name: str
  summary: str


def _feet_contact_sensor_cfg() -> ContactSensorCfg:
  return ContactSensorCfg(
    name=_FEET_CONTACT_SENSOR,
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("force",),
    reduce="netforce",
    num_slots=1,
  )


def _append_feet_contact_sensor(cfg: ManagerBasedRlEnvCfg) -> None:
  if any(sensor.name == _FEET_CONTACT_SENSOR for sensor in cfg.scene.sensors):
    return
  cfg.scene.sensors = (*cfg.scene.sensors, _feet_contact_sensor_cfg())


def _exp10_spec(base_group: int):
  return next(spec for spec in EXP10_BODY_VELOCITY_GROUP_SPECS if spec.group == base_group)


def _weight_name(weight: float) -> str:
  return f"w{abs(int(round(weight * 100))):03d}"


EXP11_BODY_VELOCITY_GROUP_SPECS: tuple[Exp11BodyVelocityGroupSpec, ...] = tuple(
  Exp11BodyVelocityGroupSpec(
    group=group_idx,
    base_exp10_group=base_group,
    foot_tilt_weight=weight,
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


def _build_exp11_body_velocity_cfg(
  spec: Exp11BodyVelocityGroupSpec, play: bool = False
) -> ManagerBasedRlEnvCfg:
  cfg = _build_exp10_body_velocity_cfg(_exp10_spec(spec.base_exp10_group), play=play)
  _append_feet_contact_sensor(cfg)
  cfg.rewards["support_foot_tilt"] = RewardTermCfg(
    func=mdp.support_foot_tilt_penalty,
    weight=spec.foot_tilt_weight,
    params={
      "sensor_name": _FEET_CONTACT_SENSOR,
      "contact_threshold": 1.0,
      "asset_cfg": SceneEntityCfg("robot", body_names=_FOOT_BODY_NAMES),
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
