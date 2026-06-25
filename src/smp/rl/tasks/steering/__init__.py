"""SMP steering tasks.

Registers ``Smp-Steering-G1``, ``Smp-Forward-G1``, and ``Smp-Upstairs-G1`` on
import.
"""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import unitree_g1_smp_ppo_runner_cfg
from smp.rl.tasks.steering.body_velocity_env_cfg import (
  BODY_VELOCITY_TASKS,
  g1_steering_double_prior_smp_env_cfg,
  g1_steering_modified_smp_env_cfg,
  g1_steering_with_stand_smp_env_cfg,
)
from smp.rl.tasks.steering.forward_env_cfg import g1_forward_smp_env_cfg
from smp.rl.tasks.steering.steering_env_cfg import g1_steering_smp_env_cfg
from smp.rl.tasks.steering.upstairs_env_cfg import g1_upstairs_smp_env_cfg


def _runner(experiment_name: str, run_name: str | None = None, init_std: float | None = None, entropy_coef: float | None = None):
  cfg = unitree_g1_smp_ppo_runner_cfg()
  cfg.experiment_name = experiment_name
  cfg.run_name = run_name or experiment_name
  if init_std is not None:
    cfg.actor.distribution_cfg["init_std"] = init_std
  if entropy_coef is not None:
    cfg.algorithm.entropy_coef = entropy_coef
  return cfg

_steering_rl = _runner("smp_steering_g1")

register_mjlab_task(
  task_id="Smp-Steering-G1",
  env_cfg=g1_steering_smp_env_cfg(play=False),
  play_env_cfg=g1_steering_smp_env_cfg(play=True),
  rl_cfg=_steering_rl,
)

_forward_rl = _runner("smp_forward_g1")

register_mjlab_task(
  task_id="Smp-Forward-G1",
  env_cfg=g1_forward_smp_env_cfg(play=False),
  play_env_cfg=g1_forward_smp_env_cfg(play=True),
  rl_cfg=_forward_rl,
)

_upstairs_rl = _runner("smp_upstairs_g1")

register_mjlab_task(
  task_id="Smp-Upstairs-G1",
  env_cfg=g1_upstairs_smp_env_cfg(play=False),
  play_env_cfg=g1_upstairs_smp_env_cfg(play=True),
  rl_cfg=_upstairs_rl,
)

_steering_modified_rl = _runner("smp_steering_modified_g1")
register_mjlab_task(
  task_id="Smp-Steering-Modified-G1",
  env_cfg=g1_steering_modified_smp_env_cfg(play=False),
  play_env_cfg=g1_steering_modified_smp_env_cfg(play=True),
  rl_cfg=_steering_modified_rl,
)

_steering_with_stand_rl = _runner("smp_steering_with_stand_g1")
register_mjlab_task(
  task_id="Smp-Steering-WithStand-G1",
  env_cfg=g1_steering_with_stand_smp_env_cfg(play=False),
  play_env_cfg=g1_steering_with_stand_smp_env_cfg(play=True),
  rl_cfg=_steering_with_stand_rl,
)

_steering_double_prior_rl = _runner("smp_steering_doubleprior_g1")
register_mjlab_task(
  task_id="Smp-Steering-DoublePrior-G1",
  env_cfg=g1_steering_double_prior_smp_env_cfg(play=False),
  play_env_cfg=g1_steering_double_prior_smp_env_cfg(play=True),
  rl_cfg=_steering_double_prior_rl,
)

_BODY_VELOCITY_RUNNER_NAMES = {
  "BodyVelocity": "smp_body_velocity_g1",
  "BodyVelocity-LafanRun": "smp_body_velocity_lafan_run_g1",
  "BodyVelocity-AmpRun": "smp_body_velocity_amp_run_g1",
  "BodyVelocity-LafanWalk": "smp_body_velocity_lafan_walk_g1",
  "BodyVelocity-LafanWalk-Matched": "smp_body_velocity_lafan_walk_matched_g1",
  "BodyVelocity-LafanWalk-Posture": "smp_body_velocity_lafan_walk_posture_g1",
  "BodyVelocity-LafanWalkRun": "smp_body_velocity_lafan_walk_run_g1",
  "BodyVelocity-AmpAll": "smp_body_velocity_amp_all_g1",
  "BodyVelocity-HandCollision": "smp_body_velocity_hand_collision_g1",
  "BodyVelocity-FootRegularized": "smp_body_velocity_foot_regularized_g1",
  "ZeroVelocity": "smp_zero_velocity_g1",
}

_BODY_VELOCITY_HIGH_NOISE = {
  "BodyVelocity-LafanRun",
  "BodyVelocity-AmpRun",
  "BodyVelocity-LafanWalk",
}

for _suffix, _env_builder in BODY_VELOCITY_TASKS.items():
  _name = _BODY_VELOCITY_RUNNER_NAMES[_suffix]
  _rl = _runner(
    _name,
    init_std=0.6 if _suffix in _BODY_VELOCITY_HIGH_NOISE else None,
    entropy_coef=0.005 if _suffix in _BODY_VELOCITY_HIGH_NOISE else None,
  )
  register_mjlab_task(
    task_id=f"Smp-{_suffix}-G1",
    env_cfg=_env_builder(play=False),
    play_env_cfg=_env_builder(play=True),
    rl_cfg=_rl,
  )

__all__ = [
  "g1_forward_smp_env_cfg",
  "g1_steering_smp_env_cfg",
  "g1_upstairs_smp_env_cfg",
  "g1_steering_modified_smp_env_cfg",
  "g1_steering_with_stand_smp_env_cfg",
  "g1_steering_double_prior_smp_env_cfg",
]
