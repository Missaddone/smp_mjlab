"""New-dev Exp13 G4--G6 with a newly trained fully mirrored SMP prior."""

from __future__ import annotations

from collections.abc import Callable

from mjlab.envs import ManagerBasedRlEnvCfg

from smp.rl.tasks.body_velocity.body_velocity_exp13_env_cfg import (
  EXP13_BODY_VELOCITY_GROUP_SPECS,
  Exp13BodyVelocityGroupSpec,
  _build_exp13_body_velocity_cfg,
)

_PRIOR_CKPT = "datasets/pretrain_ckpt/new_dev_exp13_loco_stop_static_mirrored.pt"
_GROUPS = (4, 5, 6)

EXP13_NEW_DEV_BODY_VELOCITY_GROUP_SPECS = tuple(
  spec for spec in EXP13_BODY_VELOCITY_GROUP_SPECS if spec.group in _GROUPS
)


def _build_new_dev_exp13_body_velocity_cfg(
  spec: Exp13BodyVelocityGroupSpec, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Keep Exp13 G4--G6 settings and replace only the SMP prior checkpoint."""
  cfg = _build_exp13_body_velocity_cfg(spec, play=play)
  cfg.events["init_smp_state"].params["ckpt_path"] = _PRIOR_CKPT
  return cfg


def _make_builder(
  spec: Exp13BodyVelocityGroupSpec,
) -> Callable[[bool], ManagerBasedRlEnvCfg]:
  def _builder(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _build_new_dev_exp13_body_velocity_cfg(spec, play=play)

  _builder.__name__ = f"g1_body_velocity_new_dev_exp13_group{spec.group}_smp_env_cfg"
  return _builder


EXP13_NEW_DEV_BODY_VELOCITY_BUILDERS = tuple(
  (spec.group, _make_builder(spec), spec)
  for spec in EXP13_NEW_DEV_BODY_VELOCITY_GROUP_SPECS
)
