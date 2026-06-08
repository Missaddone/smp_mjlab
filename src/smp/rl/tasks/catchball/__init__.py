"""SMP goalkeeper task — registers ``Smp-Goalkeeper-G1`` on import."""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import unitree_g1_smp_ppo_runner_cfg
from smp.rl.tasks.catchball.catchball_env_cfg import g1_catchball_smp_env_cfg

_catchball_rl = unitree_g1_smp_ppo_runner_cfg()
_catchball_rl.experiment_name = "smp_goalkeeper_g1"
_catchball_rl.run_name = "smp_goalkeeper_g1"
_catchball_rl.logger = "tensorboard"
_catchball_rl.upload_model = False

register_mjlab_task(
  task_id="Smp-Goalkeeper-G1",
  env_cfg=g1_catchball_smp_env_cfg(play=False),
  play_env_cfg=g1_catchball_smp_env_cfg(play=True),
  rl_cfg=_catchball_rl,
)

register_mjlab_task(
  task_id="Smp-CatchBall-G1",
  env_cfg=g1_catchball_smp_env_cfg(play=False),
  play_env_cfg=g1_catchball_smp_env_cfg(play=True),
  rl_cfg=_catchball_rl,
)

__all__ = [
  "g1_catchball_smp_env_cfg",
]
