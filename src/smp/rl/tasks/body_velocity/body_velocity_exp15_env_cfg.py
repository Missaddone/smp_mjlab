"""Experiment 15 staged flat-foot fine-tuning from Exp13 group 4."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.recorder_manager import RecorderTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg

from smp.rl.tasks.body_velocity import mdp
from smp.rl.tasks.body_velocity.body_velocity_exp13_env_cfg import (
  EXP13_BODY_VELOCITY_GROUP_SPECS,
  _build_exp13_body_velocity_cfg,
)

_PARENT_GROUP = 4
_STATIC_COMMAND_THRESHOLD = 0.2
_MOVING_CONTACT_THRESHOLD = 1.0
_MOVING_MIN_CONTACT_TIME = 0.06
_PHASE2_DEFAULT_STATIC_TILT_WEIGHT = 0.1
_COMMAND_NAME = "body_velocity"
_FEET_CONTACT_SENSOR = "exp15_feet_ground_contact"
_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
_DIAGNOSTIC_PATH_TEMPLATE = "logs/exp15_diagnostics/group{group:02d}_diagnostics.csv"


@dataclass(frozen=True)
class Exp15BodyVelocityGroupSpec:
  group: int
  phase: str
  iterations: int
  static_tilt_weight: float
  moving_tilt_weight: float
  run_name: str
  summary: str


def _make_specs() -> tuple[Exp15BodyVelocityGroupSpec, ...]:
  specs: list[Exp15BodyVelocityGroupSpec] = [
    Exp15BodyVelocityGroupSpec(
      group=1,
      phase="control",
      iterations=1000,
      static_tilt_weight=0.0,
      moving_tilt_weight=0.0,
      run_name="group01_control_exp13_g4_iter1000",
      summary="Continue Exp13 G4 without adding a reward, 1000 iterations",
    ),
    Exp15BodyVelocityGroupSpec(
      group=2,
      phase="control",
      iterations=3000,
      static_tilt_weight=0.0,
      moving_tilt_weight=0.0,
      run_name="group02_control_exp13_g4_iter3000",
      summary="Continue Exp13 G4 without adding a reward, 3000 iterations",
    ),
  ]

  group = 3
  for iterations in (1000, 3000):
    for weight in (0.05, 0.1, 0.2):
      specs.append(
        Exp15BodyVelocityGroupSpec(
          group=group,
          phase="static",
          iterations=iterations,
          static_tilt_weight=weight,
          moving_tilt_weight=0.0,
          run_name=(
            f"group{group:02d}_finetune_exp13_g4_static_tilt_w{weight:g}_"
            f"iter{iterations}"
          ),
          summary=(
            f"Exp13 G4 plus static double-foot tilt -{weight:g}*(t_left+t_right), "
            f"{iterations} iterations"
          ),
        )
      )
      group += 1

  specs.extend(
    (
      Exp15BodyVelocityGroupSpec(
        group=9,
        phase="moving",
        iterations=1000,
        static_tilt_weight=_PHASE2_DEFAULT_STATIC_TILT_WEIGHT,
        moving_tilt_weight=0.0,
        run_name="group09_phase2_control_selected_phase1_parent_iter1000",
        summary="Selected Phase 1 parent without adding moving tilt, 1000 iterations",
      ),
      Exp15BodyVelocityGroupSpec(
        group=10,
        phase="moving",
        iterations=1000,
        static_tilt_weight=_PHASE2_DEFAULT_STATIC_TILT_WEIGHT,
        moving_tilt_weight=0.02,
        run_name="group10_phase2_support_tilt_w002_iter1000",
        summary="Selected Phase 1 parent plus moving support tilt, 0.02, 1000 iterations",
      ),
      Exp15BodyVelocityGroupSpec(
        group=11,
        phase="moving",
        iterations=1000,
        static_tilt_weight=_PHASE2_DEFAULT_STATIC_TILT_WEIGHT,
        moving_tilt_weight=0.05,
        run_name="group11_phase2_support_tilt_w005_iter1000",
        summary="Selected Phase 1 parent plus moving support tilt, 0.05, 1000 iterations",
      ),
      Exp15BodyVelocityGroupSpec(
        group=12,
        phase="moving",
        iterations=3000,
        static_tilt_weight=_PHASE2_DEFAULT_STATIC_TILT_WEIGHT,
        moving_tilt_weight=0.02,
        run_name="group12_phase2_support_tilt_w002_iter3000",
        summary="Selected Phase 1 parent plus moving support tilt, 0.02, 3000 iterations",
      ),
      Exp15BodyVelocityGroupSpec(
        group=13,
        phase="moving",
        iterations=3000,
        static_tilt_weight=_PHASE2_DEFAULT_STATIC_TILT_WEIGHT,
        moving_tilt_weight=0.05,
        run_name="group13_phase2_support_tilt_w005_iter3000",
        summary="Selected Phase 1 parent plus moving support tilt, 0.05, 3000 iterations",
      ),
    )
  )
  return tuple(specs)


EXP15_BODY_VELOCITY_GROUP_SPECS = _make_specs()


def _parent_spec():
  return next(
    spec for spec in EXP13_BODY_VELOCITY_GROUP_SPECS if spec.group == _PARENT_GROUP
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


def _add_diagnostics(cfg: ManagerBasedRlEnvCfg, group: int) -> None:
  cfg.metrics = {
    **(cfg.metrics or {}),
    "exp15_action_delta_rms": MetricsTermCfg(func=mdp.exp15_action_delta_rms),
    "exp15_action_delta_peak": MetricsTermCfg(func=mdp.exp15_action_delta_peak),
    "exp15_action_accel_proxy_rms": MetricsTermCfg(
      func=mdp.exp15_action_accel_proxy_rms
    ),
    "exp15_action_accel_proxy_peak": MetricsTermCfg(
      func=mdp.exp15_action_accel_proxy_peak
    ),
    "exp15_joint_vel_rms": MetricsTermCfg(func=mdp.exp15_joint_vel_rms),
    "exp15_joint_vel_peak": MetricsTermCfg(func=mdp.exp15_joint_vel_peak),
    "exp15_leg_action_delta_rms": MetricsTermCfg(
      func=mdp.exp15_leg_action_delta_rms
    ),
    "exp15_leg_action_delta_peak": MetricsTermCfg(
      func=mdp.exp15_leg_action_delta_peak
    ),
    "exp15_leg_action_accel_proxy_rms": MetricsTermCfg(
      func=mdp.exp15_leg_action_accel_proxy_rms
    ),
    "exp15_leg_action_accel_proxy_peak": MetricsTermCfg(
      func=mdp.exp15_leg_action_accel_proxy_peak
    ),
    "exp15_leg_action_sign_flip_rate": MetricsTermCfg(
      func=mdp.exp15_leg_action_sign_flip_rate
    ),
    "exp15_leg_joint_vel_rms": MetricsTermCfg(func=mdp.exp15_leg_joint_vel_rms),
    "exp15_leg_joint_vel_peak": MetricsTermCfg(func=mdp.exp15_leg_joint_vel_peak),
    "exp15_foot_action_delta_rms": MetricsTermCfg(
      func=mdp.exp15_foot_action_delta_rms
    ),
    "exp15_foot_action_delta_peak": MetricsTermCfg(
      func=mdp.exp15_foot_action_delta_peak
    ),
    "exp15_foot_action_accel_proxy_rms": MetricsTermCfg(
      func=mdp.exp15_foot_action_accel_proxy_rms
    ),
    "exp15_foot_action_accel_proxy_peak": MetricsTermCfg(
      func=mdp.exp15_foot_action_accel_proxy_peak
    ),
    "exp15_foot_action_sign_flip_rate": MetricsTermCfg(
      func=mdp.exp15_foot_action_sign_flip_rate
    ),
    "exp15_foot_joint_vel_rms": MetricsTermCfg(func=mdp.exp15_foot_joint_vel_rms),
    "exp15_foot_joint_vel_peak": MetricsTermCfg(
      func=mdp.exp15_foot_joint_vel_peak
    ),
  }
  cfg.recorders = {
    **(cfg.recorders or {}),
    "exp15_diagnostics_csv": RecorderTermCfg(
      func=mdp.Exp15DiagnosticsRecorder,
      params={
        "path": _DIAGNOSTIC_PATH_TEMPLATE.format(group=group),
        "flush_interval_steps": 50,
      },
    )
  }


def _build_exp15_body_velocity_cfg(
  spec: Exp15BodyVelocityGroupSpec, play: bool = False
) -> ManagerBasedRlEnvCfg:
  cfg = _build_exp13_body_velocity_cfg(_parent_spec(), play=play)
  _add_diagnostics(cfg, spec.group)

  feet_cfg = SceneEntityCfg("robot", body_names=_FOOT_BODY_NAMES)
  if spec.static_tilt_weight != 0.0:
    for foot_index, foot_name in enumerate(("left", "right")):
      cfg.rewards[f"exp15_static_{foot_name}_foot_tilt"] = RewardTermCfg(
        func=mdp.static_foot_tilt_penalty,
        weight=-spec.static_tilt_weight,
        params={
          "command_name": _COMMAND_NAME,
          "foot_index": foot_index,
          "command_zero_threshold": _STATIC_COMMAND_THRESHOLD,
          "asset_cfg": feet_cfg,
        },
      )

  if spec.moving_tilt_weight != 0.0:
    _append_feet_contact_sensor(cfg)
    cfg.rewards["exp15_moving_support_foot_tilt"] = RewardTermCfg(
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
  return cfg


def _make_builder(
  spec: Exp15BodyVelocityGroupSpec,
) -> Callable[[bool], ManagerBasedRlEnvCfg]:
  def _builder(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _build_exp15_body_velocity_cfg(spec, play=play)

  _builder.__name__ = f"g1_body_velocity_exp15_group{spec.group}_smp_env_cfg"
  return _builder


EXP15_BODY_VELOCITY_BUILDERS = tuple(
  (spec.group, _make_builder(spec), spec)
  for spec in EXP15_BODY_VELOCITY_GROUP_SPECS
)
