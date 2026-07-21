"""SMP body-velocity tasks."""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import unitree_g1_smp_ppo_runner_cfg
from smp.rl.tasks.body_velocity.body_velocity_env_cfg import (
  g1_body_velocity_smp_env_cfg,
)
from smp.rl.tasks.body_velocity.body_velocity_exp4_group1_env_cfg import (
  g1_body_velocity_exp4_group1_smp_env_cfg,
)
from smp.rl.tasks.body_velocity.body_velocity_exp4_group2_env_cfg import (
  g1_body_velocity_exp4_group2_smp_env_cfg,
)
from smp.rl.tasks.body_velocity.body_velocity_exp4_group3_env_cfg import (
  g1_body_velocity_exp4_group3_smp_env_cfg,
)
from smp.rl.tasks.body_velocity.body_velocity_exp4_group4_env_cfg import (
  g1_body_velocity_exp4_group4_smp_env_cfg,
)
from smp.rl.tasks.body_velocity.body_velocity_exp4_group5_env_cfg import (
  g1_body_velocity_exp4_group5_smp_env_cfg,
)
from smp.rl.tasks.body_velocity.body_velocity_exp4_group6_env_cfg import (
  g1_body_velocity_exp4_group6_smp_env_cfg,
)
from smp.rl.tasks.body_velocity.body_velocity_exp4_group7_env_cfg import (
  g1_body_velocity_exp4_group7_smp_env_cfg,
)
from smp.rl.tasks.body_velocity.body_velocity_exp10_env_cfg import (
  EXP10_BODY_VELOCITY_BUILDERS,
)
from smp.rl.tasks.body_velocity.body_velocity_exp11_env_cfg import (
  EXP11_BODY_VELOCITY_BUILDERS,
)
from smp.rl.tasks.body_velocity.body_velocity_exp12_env_cfg import (
  EXP12_BODY_VELOCITY_BUILDERS,
)


def _runner(experiment_name: str, run_name: str | None = None):
  cfg = unitree_g1_smp_ppo_runner_cfg()
  cfg.experiment_name = experiment_name
  cfg.run_name = run_name or experiment_name
  return cfg

_body_velocity_rl = _runner("smp_body_velocity_g1")

register_mjlab_task(
  task_id="Smp-BodyVelocity-G1",
  env_cfg=g1_body_velocity_smp_env_cfg(play=False),
  play_env_cfg=g1_body_velocity_smp_env_cfg(play=True),
  rl_cfg=_body_velocity_rl,
)

_EXP4_GROUPS = (
  (1, g1_body_velocity_exp4_group1_smp_env_cfg),
  (2, g1_body_velocity_exp4_group2_smp_env_cfg),
  (3, g1_body_velocity_exp4_group3_smp_env_cfg),
  (4, g1_body_velocity_exp4_group4_smp_env_cfg),
  (5, g1_body_velocity_exp4_group5_smp_env_cfg),
  (6, g1_body_velocity_exp4_group6_smp_env_cfg),
  (7, g1_body_velocity_exp4_group7_smp_env_cfg),
)

for _group, _env_builder in _EXP4_GROUPS:
  _rl = _runner(
    "smp_exp4_body_velocity_ablation",
    run_name=f"group{_group}",
  )
  register_mjlab_task(
    task_id=f"Smp-BodyVelocity-Exp4-Group{_group}-G1",
    env_cfg=_env_builder(play=False),
    play_env_cfg=_env_builder(play=True),
    rl_cfg=_rl,
  )

for _group, _env_builder, _spec in EXP10_BODY_VELOCITY_BUILDERS:
  _rl = _runner(
    "smp_exp10_static_steering_body_velocity",
    run_name=_spec.run_name,
  )
  register_mjlab_task(
    task_id=f"Smp-BodyVelocity-Exp10-Group{_group}-G1",
    env_cfg=_env_builder(play=False),
    play_env_cfg=_env_builder(play=True),
    rl_cfg=_rl,
  )

for _group, _env_builder, _spec in EXP11_BODY_VELOCITY_BUILDERS:
  _rl = _runner(
    "smp_exp11_body_velocity_foot_tilt",
    run_name=_spec.run_name,
  )
  register_mjlab_task(
    task_id=f"Smp-BodyVelocity-Exp11-Group{_group}-G1",
    env_cfg=_env_builder(play=False),
    play_env_cfg=_env_builder(play=True),
    rl_cfg=_rl,
  )

for _group, _env_builder, _spec in EXP12_BODY_VELOCITY_BUILDERS:
  _rl = _runner(
    "smp_exp12_body_velocity_theme_prior",
    run_name=_spec.run_name,
  )
  register_mjlab_task(
    task_id=f"Smp-BodyVelocity-Exp12-Group{_group}-G1",
    env_cfg=_env_builder(play=False),
    play_env_cfg=_env_builder(play=True),
    rl_cfg=_rl,
  )

__all__ = [
  "EXP10_BODY_VELOCITY_BUILDERS",
  "EXP11_BODY_VELOCITY_BUILDERS",
  "EXP12_BODY_VELOCITY_BUILDERS",
  "g1_body_velocity_exp4_group1_smp_env_cfg",
  "g1_body_velocity_exp4_group2_smp_env_cfg",
  "g1_body_velocity_exp4_group3_smp_env_cfg",
  "g1_body_velocity_exp4_group4_smp_env_cfg",
  "g1_body_velocity_exp4_group5_smp_env_cfg",
  "g1_body_velocity_exp4_group6_smp_env_cfg",
  "g1_body_velocity_exp4_group7_smp_env_cfg",
  "g1_body_velocity_smp_env_cfg",
]
