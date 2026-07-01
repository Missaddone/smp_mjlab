"""SMP goalkeeper task — registers ``Smp-Goalkeeper-G1`` on import."""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import unitree_g1_smp_ppo_runner_cfg
from smp.rl.tasks.catchball.catchball_env_cfg import g1_catchball_smp_env_cfg

def _runner(experiment_name: str):
  cfg = unitree_g1_smp_ppo_runner_cfg()
  cfg.experiment_name = experiment_name
  cfg.run_name = experiment_name
  cfg.logger = "wandb"
  cfg.wandb_project = experiment_name
  cfg.upload_model = False
  return cfg


_goalkeeper_rl = _runner("smp_goalkeeper_g1")
_catchball_rl = _runner("smp_catchball_g1")

register_mjlab_task(
  task_id="Smp-Goalkeeper-G1",
  env_cfg=g1_catchball_smp_env_cfg(play=False),
  play_env_cfg=g1_catchball_smp_env_cfg(play=True),
  rl_cfg=_goalkeeper_rl,
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
