"""Experiment 13 body-velocity moving-reward product-mix ablations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg

from smp.rl.tasks.body_velocity.body_velocity_exp10_env_cfg import (
  EXP10_BODY_VELOCITY_GROUP_SPECS,
  _build_exp10_body_velocity_cfg,
  set_body_velocity_moving_product_mix,
)


@dataclass(frozen=True)
class Exp13MovingRewardSpec:
  key: str
  product_weight: float
  linear_weight: float
  yaw_weight: float


@dataclass(frozen=True)
class Exp13BodyVelocityGroupSpec:
  group: int
  base_exp10_group: int
  reward: Exp13MovingRewardSpec
  run_name: str
  summary: str


_MOVING_REWARD_SPECS: tuple[Exp13MovingRewardSpec, ...] = (
  Exp13MovingRewardSpec("mix050_025_025", 0.5, 0.25, 0.25),
  Exp13MovingRewardSpec("mix060_020_020", 0.6, 0.2, 0.2),
  Exp13MovingRewardSpec("mix070_015_015", 0.7, 0.15, 0.15),
)


def _exp10_spec(base_group: int):
  return next(spec for spec in EXP10_BODY_VELOCITY_GROUP_SPECS if spec.group == base_group)


EXP13_BODY_VELOCITY_GROUP_SPECS: tuple[Exp13BodyVelocityGroupSpec, ...] = tuple(
  Exp13BodyVelocityGroupSpec(
    group=group,
    base_exp10_group=base_group,
    reward=reward,
    run_name=(
      f"group{group:02d}_from_exp10_g{base_group}_moving_{reward.key}"
    ),
    summary=(
      f"Exp10 group{base_group} unchanged except moving reward "
      f"{reward.product_weight}*r_l*r_y + {reward.linear_weight}*r_l + "
      f"{reward.yaw_weight}*r_y"
    ),
  )
  for group, (base_group, reward) in enumerate(
    (
      *((14, reward) for reward in _MOVING_REWARD_SPECS),
      *((15, reward) for reward in _MOVING_REWARD_SPECS),
    ),
    start=1,
  )
)


def _build_exp13_body_velocity_cfg(
  spec: Exp13BodyVelocityGroupSpec, play: bool = False
) -> ManagerBasedRlEnvCfg:
  cfg = _build_exp10_body_velocity_cfg(_exp10_spec(spec.base_exp10_group), play=play)
  set_body_velocity_moving_product_mix(
    cfg,
    product_weight=spec.reward.product_weight,
    linear_weight=spec.reward.linear_weight,
    yaw_weight=spec.reward.yaw_weight,
  )
  return cfg


def _make_builder(
  spec: Exp13BodyVelocityGroupSpec,
) -> Callable[[bool], ManagerBasedRlEnvCfg]:
  def _builder(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _build_exp13_body_velocity_cfg(spec, play=play)

  _builder.__name__ = f"g1_body_velocity_exp13_group{spec.group}_smp_env_cfg"
  return _builder


EXP13_BODY_VELOCITY_BUILDERS = tuple(
  (spec.group, _make_builder(spec), spec)
  for spec in EXP13_BODY_VELOCITY_GROUP_SPECS
)
