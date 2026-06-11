"""SMP steering tasks.

Registers ``Smp-Steering-G1``, ``Smp-Forward-G1``, and ``Smp-Upstairs-G1`` on
import.
"""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import unitree_g1_smp_ppo_runner_cfg
from smp.rl.tasks.steering.forward_env_cfg import g1_forward_smp_env_cfg
from smp.rl.tasks.steering.steering_env_cfg import g1_steering_smp_env_cfg
from smp.rl.tasks.steering.upstairs_env_cfg import g1_upstairs_smp_env_cfg

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

_upstairs_rl = unitree_g1_smp_ppo_runner_cfg()
_upstairs_rl.experiment_name = "smp_upstairs_g1"
_upstairs_rl.run_name = "smp_upstairs_g1"

register_mjlab_task(
  task_id="Smp-Upstairs-G1",
  env_cfg=g1_upstairs_smp_env_cfg(play=False),
  play_env_cfg=g1_upstairs_smp_env_cfg(play=True),
  rl_cfg=_upstairs_rl,
)

__all__ = [
  "g1_forward_smp_env_cfg",
  "g1_steering_smp_env_cfg",
  "g1_upstairs_smp_env_cfg",
]
