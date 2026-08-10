"""New-dev Experiment 14: strict, short, isolated flat-foot fine-tunes.

Each group starts from one newly trained Exp13 base policy.  The reward terms
are deliberately mutually exclusive: no-extra-reward control, moving support
foot tilt, or static double-foot tilt.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg

from smp.rl.tasks.body_velocity import mdp
from smp.rl.tasks.body_velocity.body_velocity_exp13_env_cfg import (
  EXP13_BODY_VELOCITY_GROUP_SPECS,
  _build_exp13_body_velocity_cfg,
)

_PARENT_GROUPS = (4, 5, 6)
_STATIC_COMMAND_THRESHOLD = 0.2
_MOVING_CONTACT_THRESHOLD = 1.0
_MOVING_MIN_CONTACT_TIME = 0.06
_FLATFOOT_WEIGHT = 0.1
_COMMAND_NAME = "body_velocity"
_FEET_CONTACT_SENSOR = "new_dev_exp14_feet_ground_contact"
_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")


@dataclass(frozen=True)
class NewDevExp14BodyVelocityGroupSpec:
  group: int
  parent_group: int
  kind: str
  iterations: int
  moving_tilt_weight: float
  static_tilt_weight: float
  run_name: str
  summary: str


def _make_specs() -> tuple[NewDevExp14BodyVelocityGroupSpec, ...]:
  specs: list[NewDevExp14BodyVelocityGroupSpec] = []
  group = 1
  for kind, moving_weight, static_weight in (
    ("control", 0.0, 0.0),
    ("moving", _FLATFOOT_WEIGHT, 0.0),
    ("static", 0.0, _FLATFOOT_WEIGHT),
  ):
    for parent_group in _PARENT_GROUPS:
      suffix = {
        "control": "control",
        "moving": "moving_debounced_support_tilt_w010",
        "static": "static_double_foot_tilt_w010",
      }[kind]
      summary = {
        "control": f"Continue new-dev Exp13 G{parent_group} without extra reward",
        "moving": (
          f"New-dev Exp13 G{parent_group} plus moving debounced support-foot "
          "tilt, F>1N and contact >0.06s"
        ),
        "static": (
          f"New-dev Exp13 G{parent_group} plus static left/right foot tilt "
          "when command norm <=0.2"
        ),
      }[kind]
      specs.append(
        NewDevExp14BodyVelocityGroupSpec(
          group=group,
          parent_group=parent_group,
          kind=kind,
          iterations=100,
          moving_tilt_weight=moving_weight,
          static_tilt_weight=static_weight,
          run_name=f"group{group:02d}_from_newdev_exp13_g{parent_group}_{suffix}",
          summary=summary,
        )
      )
      group += 1
  return tuple(specs)


EXP14_NEW_DEV_BODY_VELOCITY_GROUP_SPECS = _make_specs()


def _parent_spec(parent_group: int):
  return next(
    spec for spec in EXP13_BODY_VELOCITY_GROUP_SPECS if spec.group == parent_group
  )


def _append_feet_contact_sensor(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.scene.sensors = (
    *cfg.scene.sensors,
    ContactSensorCfg(
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
    ),
  )


def _build_exp14_new_dev_body_velocity_cfg(
  spec: NewDevExp14BodyVelocityGroupSpec, play: bool = False
) -> ManagerBasedRlEnvCfg:
  cfg = _build_exp13_body_velocity_cfg(_parent_spec(spec.parent_group), play=play)
  feet_cfg = SceneEntityCfg("robot", body_names=_FOOT_BODY_NAMES)

  if spec.moving_tilt_weight != 0.0:
    _append_feet_contact_sensor(cfg)
    cfg.rewards["new_dev_exp14_moving_support_foot_tilt"] = RewardTermCfg(
      func=mdp.moving_debounced_support_foot_tilt_penalty,
      weight=-spec.moving_tilt_weight,
      params={
        "command_name": _COMMAND_NAME,
        "sensor_name": _FEET_CONTACT_SENSOR,
        "contact_threshold": _MOVING_CONTACT_THRESHOLD,
        "min_contact_time": _MOVING_MIN_CONTACT_TIME,
        "command_threshold": _STATIC_COMMAND_THRESHOLD,
        "asset_cfg": feet_cfg,
      },
    )

  if spec.static_tilt_weight != 0.0:
    for foot_index, foot_name in enumerate(("left", "right")):
      cfg.rewards[f"new_dev_exp14_static_{foot_name}_foot_tilt"] = RewardTermCfg(
        func=mdp.static_foot_tilt_penalty,
        weight=-spec.static_tilt_weight,
        params={
          "command_name": _COMMAND_NAME,
          "foot_index": foot_index,
          "command_zero_threshold": _STATIC_COMMAND_THRESHOLD,
          "asset_cfg": feet_cfg,
        },
      )
  return cfg


def _make_builder(
  spec: NewDevExp14BodyVelocityGroupSpec,
) -> Callable[[bool], ManagerBasedRlEnvCfg]:
  def _builder(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _build_exp14_new_dev_body_velocity_cfg(spec, play=play)

  _builder.__name__ = f"g1_body_velocity_new_dev_exp14_group{spec.group}_smp_env_cfg"
  return _builder


EXP14_NEW_DEV_BODY_VELOCITY_BUILDERS = tuple(
  (spec.group, _make_builder(spec), spec)
  for spec in EXP14_NEW_DEV_BODY_VELOCITY_GROUP_SPECS
)
