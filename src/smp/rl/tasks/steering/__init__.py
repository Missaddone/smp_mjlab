"""SMP steering tasks.

Registers ``Smp-Steering-G1``, ``Smp-Forward-G1``, and ``Smp-Upstairs-G1`` on
import.
"""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import unitree_g1_smp_ppo_runner_cfg
from smp.rl.tasks.steering.body_forward_backward_env_cfg import (
  g1_body_forward_backward_smp_env_cfg,
)
from smp.rl.tasks.steering.body_velocity_env_cfg import (
  BODY_VELOCITY_TASKS,
  g1_steering_modified_smp_env_cfg,
)
from smp.rl.tasks.steering.forward_env_cfg import (
  g1_forward_backward_smp_env_cfg,
  g1_forward_smp_env_cfg,
)
from smp.rl.tasks.steering.steering_env_cfg import g1_steering_smp_env_cfg
from smp.rl.tasks.steering.upstairs_env_cfg import g1_upstairs_smp_env_cfg


def _runner(experiment_name: str, run_name: str | None = None, init_std: float | None = None, entropy_coef: float | None = None):
  cfg = unitree_g1_smp_ppo_runner_cfg()
  cfg.experiment_name = experiment_name
  cfg.run_name = run_name or experiment_name
  if init_std is not None:
    assert cfg.actor.distribution_cfg is not None
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

_forward_backward_rl = _runner("smp_forward_backward_g1")

register_mjlab_task(
  task_id="Smp-ForwardBackward-G1",
  env_cfg=g1_forward_backward_smp_env_cfg(play=False),
  play_env_cfg=g1_forward_backward_smp_env_cfg(play=True),
  rl_cfg=_forward_backward_rl,
)

_body_forward_backward_rl = _runner(
  "smp_body_forward_backward_g1",
  init_std=0.6,
  entropy_coef=0.005,
)

register_mjlab_task(
  task_id="Smp-BodyForwardBackward-G1",
  env_cfg=g1_body_forward_backward_smp_env_cfg(play=False),
  play_env_cfg=g1_body_forward_backward_smp_env_cfg(play=True),
  rl_cfg=_body_forward_backward_rl,
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

_BODY_VELOCITY_RUNNER_NAMES = {
  "BodyVelocity": "smp_body_velocity_g1",
  "BodyVelocity-UnitreeRef": "smp_body_velocity_unitree_ref_g1",
  "BodyVelocity-Sum": "smp_body_velocity_sum_g1",
  "BodyVelocity-Walk": "smp_body_velocity_walk_g1",
  "BodyVelocity-Run": "smp_body_velocity_run_g1",
  "BodyVelocity-FootRegularized": "smp_body_velocity_foot_regularized_g1",
}

_BODY_VELOCITY_HIGH_NOISE = {
  "BodyVelocity-Walk",
  "BodyVelocity-Run",
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
    env_cfg=_env_builder(False),
    play_env_cfg=_env_builder(True),
    rl_cfg=_rl,
  )

__all__ = [
  "g1_forward_smp_env_cfg",
  "g1_forward_backward_smp_env_cfg",
  "g1_body_forward_backward_smp_env_cfg",
  "g1_steering_smp_env_cfg",
  "g1_upstairs_smp_env_cfg",
  "g1_steering_modified_smp_env_cfg",
]
