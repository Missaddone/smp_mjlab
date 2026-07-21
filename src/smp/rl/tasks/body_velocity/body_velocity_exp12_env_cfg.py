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
)

_BASE_EXP10_GROUP = 14
_FEET_CONTACT_SENSOR = "exp12_feet_ground_contact"
_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
_FOOT_TILT_WEIGHTS = (-0.05, -0.1, -0.2, -0.3)


@dataclass(frozen=True)
class Exp12BodyVelocityGroupSpec:
  group: int
  style: str
  prior_ckpt: str
  foot_tilt_weight: float
  run_name: str
  summary: str


def _exp10_group14_spec():
  return next(spec for spec in EXP10_BODY_VELOCITY_GROUP_SPECS if spec.group == _BASE_EXP10_GROUP)


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


def _weight_name(weight: float) -> str:
  return f"w{abs(int(round(weight * 100))):03d}"


_STYLE_PRIORS: tuple[tuple[str, str], ...] = (
  ("male", "datasets/pretrain_ckpt/exp12_theme_male.pt"),
  ("female", "datasets/pretrain_ckpt/exp12_theme_female.pt"),
  ("children", "datasets/pretrain_ckpt/exp12_theme_children.pt"),
)


EXP12_BODY_VELOCITY_GROUP_SPECS: tuple[Exp12BodyVelocityGroupSpec, ...] = tuple(
  Exp12BodyVelocityGroupSpec(
    group=group,
    style=style,
    prior_ckpt=prior_ckpt,
    foot_tilt_weight=weight,
    run_name=(
      f"group{group:02d}_theme_{style}_exp10_g14_"
      f"support_foot_tilt_{_weight_name(weight)}"
    ),
    summary=(
      f"Exp10 group14 config, {style} theme prior, "
      f"{weight}*support_foot_tilt_penalty"
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


def _build_exp12_body_velocity_cfg(
  spec: Exp12BodyVelocityGroupSpec, play: bool = False
) -> ManagerBasedRlEnvCfg:
  cfg = _build_exp10_body_velocity_cfg(_exp10_group14_spec(), play=play)
  cfg.events["init_smp_state"].params["ckpt_path"] = spec.prior_ckpt
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
