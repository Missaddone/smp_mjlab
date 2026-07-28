"""Experiment 14: Exp13 G5 flat-foot and contact-duty fine-tunes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg

from smp.rl.rewards import body_velocity_task_smp_product_with_multiplier
from smp.rl.tasks.body_velocity import mdp
from smp.rl.tasks.body_velocity.body_velocity_exp13_env_cfg import (
  EXP13_BODY_VELOCITY_GROUP_SPECS,
  _build_exp13_body_velocity_cfg,
)

_PARENT_GROUP = 5
_FEET_CONTACT_SENSOR = "exp14_feet_ground_contact"
_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
_STATIC_COMMAND_THRESHOLD = 0.2
_GAIT_CONTACT_THRESHOLD = 1.0
_GAIT_WINDOW_SECONDS = 3.0
_GAIT_EXP_SCALE = 2.0


@dataclass(frozen=True)
class Exp14BodyVelocityGroupSpec:
  group: int
  kind: str
  iterations: int
  strength: float | None
  run_name: str
  summary: str


def _make_specs() -> tuple[Exp14BodyVelocityGroupSpec, ...]:
  specs: list[Exp14BodyVelocityGroupSpec] = []
  group = 1
  for strength in (0.2, 0.5, 1.0):
    for iterations in (3000, 6000):
      specs.append(
        Exp14BodyVelocityGroupSpec(
          group=group,
          kind="static_tilt_additive",
          iterations=iterations,
          strength=strength,
          run_name=(
            f"group{group:02d}_finetune_exp13_g5_static_tilt_add_w{strength:g}_"
            f"iter{iterations}"
          ),
          summary=(
            "Fine-tune Exp13 G5; static only: "
            f"R=R0-{strength:g}*(t_left+t_right); {iterations} iterations"
          ),
        )
      )
      group += 1

  for iterations in (3000, 6000):
    specs.append(
      Exp14BodyVelocityGroupSpec(
        group=group,
        kind="static_tilt_product",
        iterations=iterations,
        strength=None,
        run_name=f"group{group:02d}_finetune_exp13_g5_static_tilt_product_iter{iterations}",
        summary=(
          "Fine-tune Exp13 G5; static only: "
          f"R=R0*exp(-t_left)*exp(-t_right); {iterations} iterations"
        ),
      )
    )
    group += 1

  for iterations in (3000, 6000):
    specs.append(
      Exp14BodyVelocityGroupSpec(
        group=group,
        kind="gait_duty_product",
        iterations=iterations,
        strength=_GAIT_EXP_SCALE,
        run_name=(
          f"group{group:02d}_finetune_exp13_g5_gait_duty_exp_k{_GAIT_EXP_SCALE:g}_"
          f"iter{iterations}"
        ),
        summary=(
          "Fine-tune Exp13 G5; moving only: "
          f"R=R0*exp(-{_GAIT_EXP_SCALE:g}*duty_error), 3s/1N; {iterations} iterations"
        ),
      )
    )
    group += 1

  for strength in (0.05, 0.1, 0.2):
    for iterations in (3000, 6000):
      specs.append(
        Exp14BodyVelocityGroupSpec(
          group=group,
          kind="gait_duty_additive",
          iterations=iterations,
          strength=strength,
          run_name=(
            f"group{group:02d}_finetune_exp13_g5_gait_duty_add_w{strength:g}_"
            f"iter{iterations}"
          ),
          summary=(
            "Fine-tune Exp13 G5; moving only: "
            f"R=R0-{strength:g}*duty_error, 3s/1N; {iterations} iterations"
          ),
        )
      )
      group += 1
  return tuple(specs)


EXP14_BODY_VELOCITY_GROUP_SPECS = _make_specs()


def _parent_spec():
  return next(spec for spec in EXP13_BODY_VELOCITY_GROUP_SPECS if spec.group == _PARENT_GROUP)


def _append_feet_contact_sensor(cfg: ManagerBasedRlEnvCfg) -> None:
  if any(sensor.name == _FEET_CONTACT_SENSOR for sensor in cfg.scene.sensors):
    return
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
      fields=("force",),
      reduce="netforce",
      num_slots=1,
      track_air_time=False,
    ),
  )


def _replace_task_smp_product_with_multiplier(
  cfg: ManagerBasedRlEnvCfg,
  multiplier_func: Callable,
  multiplier_params: dict,
) -> None:
  reward_cfg = cfg.rewards["task_smp_product"]
  reward_cfg.func = body_velocity_task_smp_product_with_multiplier
  reward_cfg.params = {
    **reward_cfg.params,
    "multiplier_func": multiplier_func,
    "multiplier_params": multiplier_params,
  }


def _build_exp14_body_velocity_cfg(
  spec: Exp14BodyVelocityGroupSpec, play: bool = False
) -> ManagerBasedRlEnvCfg:
  cfg = _build_exp13_body_velocity_cfg(_parent_spec(), play=play)
  feet_cfg = SceneEntityCfg("robot", body_names=_FOOT_BODY_NAMES)
  static_params = {
    "command_name": "body_velocity",
    "command_zero_threshold": _STATIC_COMMAND_THRESHOLD,
    "asset_cfg": feet_cfg,
  }
  duty_params = {
    "command_name": "body_velocity",
    "sensor_name": _FEET_CONTACT_SENSOR,
    "contact_threshold": _GAIT_CONTACT_THRESHOLD,
    "window_seconds": _GAIT_WINDOW_SECONDS,
    "command_threshold": _STATIC_COMMAND_THRESHOLD,
  }

  if spec.kind == "static_tilt_additive":
    assert spec.strength is not None
    for foot_index, foot_name in enumerate(("left", "right")):
      cfg.rewards[f"static_{foot_name}_foot_tilt"] = RewardTermCfg(
        func=mdp.static_foot_tilt_penalty,
        weight=-spec.strength,
        params={**static_params, "foot_index": foot_index},
      )
  elif spec.kind == "static_tilt_product":
    _replace_task_smp_product_with_multiplier(
      cfg,
      mdp.static_double_foot_tilt_multiplier,
      static_params,
    )
  elif spec.kind == "gait_duty_product":
    _append_feet_contact_sensor(cfg)
    _replace_task_smp_product_with_multiplier(
      cfg,
      mdp.moving_foot_contact_duty_multiplier,
      {**duty_params, "exp_scale": spec.strength},
    )
  elif spec.kind == "gait_duty_additive":
    assert spec.strength is not None
    _append_feet_contact_sensor(cfg)
    cfg.rewards["moving_foot_contact_duty"] = RewardTermCfg(
      func=mdp.moving_foot_contact_duty_error,
      weight=-spec.strength,
      params=duty_params,
    )
  else:
    raise ValueError(f"Unsupported Exp14 reward kind: {spec.kind}")
  return cfg


def _make_builder(
  spec: Exp14BodyVelocityGroupSpec,
) -> Callable[[bool], ManagerBasedRlEnvCfg]:
  def _builder(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _build_exp14_body_velocity_cfg(spec, play=play)

  _builder.__name__ = f"g1_body_velocity_exp14_group{spec.group}_smp_env_cfg"
  return _builder


EXP14_BODY_VELOCITY_BUILDERS = tuple(
  (spec.group, _make_builder(spec), spec) for spec in EXP14_BODY_VELOCITY_GROUP_SPECS
)
