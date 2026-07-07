"""SMP steering tasks — registers ``Smp-Steering-G1`` and ``Smp-Forward-G1``
on import."""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import unitree_g1_smp_ppo_runner_cfg
from smp.rl.tasks.steering.forward_env_cfg import g1_forward_smp_env_cfg
from smp.rl.tasks.steering.forward_exp6_group4_env_cfg import (
  g1_forward_exp6_group4_smp_env_cfg,
)
from smp.rl.tasks.steering.forward_exp6_group5_env_cfg import (
  g1_forward_exp6_group5_smp_env_cfg,
)
from smp.rl.tasks.steering.forward_exp6_group6_env_cfg import (
  g1_forward_exp6_group6_smp_env_cfg,
)
from smp.rl.tasks.steering.steering_env_cfg import g1_steering_smp_env_cfg

_steering_rl = unitree_g1_smp_ppo_runner_cfg()
_steering_rl.experiment_name = "smp_steering_g1"
_steering_rl.run_name = "smp_steering_g1"

register_mjlab_task(
  task_id="Smp-Steering-G1",
  env_cfg=g1_steering_smp_env_cfg(play=False),
  play_env_cfg=g1_steering_smp_env_cfg(play=True),
  rl_cfg=_steering_rl,
)

_forward_rl = unitree_g1_smp_ppo_runner_cfg()
_forward_rl.experiment_name = "smp_forward_g1"
_forward_rl.run_name = "smp_forward_g1"

register_mjlab_task(
  task_id="Smp-Forward-G1",
  env_cfg=g1_forward_smp_env_cfg(play=False),
  play_env_cfg=g1_forward_smp_env_cfg(play=True),
  rl_cfg=_forward_rl,
)

_EXP6_FORWARD_GROUPS = (
  (4, g1_forward_exp6_group4_smp_env_cfg),
  (5, g1_forward_exp6_group5_smp_env_cfg),
  (6, g1_forward_exp6_group6_smp_env_cfg),
)

for _group, _env_builder in _EXP6_FORWARD_GROUPS:
  _rl = unitree_g1_smp_ppo_runner_cfg()
  _rl.experiment_name = "smp_exp6_forward_stop"
  _rl.run_name = f"group{_group}"
  register_mjlab_task(
    task_id=f"Smp-Forward-Exp6-Group{_group}-G1",
    env_cfg=_env_builder(play=False),
    play_env_cfg=_env_builder(play=True),
    rl_cfg=_rl,
  )

__all__ = [
  "g1_forward_exp6_group4_smp_env_cfg",
  "g1_forward_exp6_group5_smp_env_cfg",
  "g1_forward_exp6_group6_smp_env_cfg",
  "g1_forward_smp_env_cfg",
  "g1_steering_smp_env_cfg",
]
