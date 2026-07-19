"""Experiment 9 forward configs: Exp7 scene with prior/reward grid."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg

from smp.rl.rewards import task_smp_product
from smp.rl.tasks.steering import mdp
from smp.rl.tasks.steering.forward_exp6_group7_env_cfg import (
  g1_forward_exp6_group7_smp_env_cfg,
)

_ZERO_THRESHOLD = 0.2
_VEL_ERR_SCALE = 0.5
_ROOT_VEL_EXP_SCALE = 2.0
_JOINT_VEL_EXP_SCALE = 0.02
_ROOT_ANG_EXP_SCALE = 1.0
_UPPER_JOINT_VEL_EXP_SCALE = 0.05
_ACTION_SMOOTH_EXP_SCALE = 0.25
_DEAD_ZONE_SPEED = 1.0


@dataclass(frozen=True)
class Exp9PriorSpec:
  index: int
  name: str
  ckpt_path: str
  summary: str


@dataclass(frozen=True)
class Exp9RewardSpec:
  key: str
  name: str
  summary: str
  task_terms: tuple


@dataclass(frozen=True)
class Exp9GroupSpec:
  group: int
  prior: Exp9PriorSpec
  reward: Exp9RewardSpec
  run_name: str
  dead_zone_speed: float = 0.0


EXP9_PRIORS: tuple[Exp9PriorSpec, ...] = (
  Exp9PriorSpec(
    index=1,
    name="forward_walk_to_stop",
    ckpt_path="datasets/pretrain_ckpt/exp9_prior1_forward_walk_to_stop.pt",
    summary="forward + walk_to_stop",
  ),
  Exp9PriorSpec(
    index=2,
    name="forward_w2s_s2w",
    ckpt_path="datasets/pretrain_ckpt/exp9_prior2_forward_w2s_s2w.pt",
    summary="forward + walk_to_stop + stop_to_walk",
  ),
  Exp9PriorSpec(
    index=3,
    name="forward_w2s_s2w_stop_static",
    ckpt_path="datasets/pretrain_ckpt/exp9_prior3_forward_w2s_s2w_stop_static.pt",
    summary="forward + walk_to_stop + stop_to_walk + stop_static",
  ),
  Exp9PriorSpec(
    index=4,
    name="forward_stop_static",
    ckpt_path="datasets/pretrain_ckpt/exp9_prior4_forward_stop_static.pt",
    summary="forward + stop_static",
  ),
)

EXP9_PRIOR5 = Exp9PriorSpec(
  index=5,
  name="forward_stop_static_low_walk",
  ckpt_path="datasets/pretrain_ckpt/exp9_prior5_forward_stop_static_low_walk.pt",
  summary="forward + stop_static + g1_low_walk",
)


def _base_velocity_term() -> tuple:
  return (
    mdp.steering_target_velocity,
    1.0,
    {"command_name": "steering", "vel_err_scale": _VEL_ERR_SCALE},
  )


def _standstill_joint_vel_penalty_term() -> tuple:
  return (
    mdp.stand_still_vel,
    -0.02,
    {
      "command_name": "steering",
      "zero_threshold": _ZERO_THRESHOLD,
      "joint_vel_scale": 1.0,
    },
  )


EXP9_REWARDS: tuple[Exp9RewardSpec, ...] = (
  Exp9RewardSpec(
    key="A",
    name="exp7g4_joint_vel_penalty",
    summary="A: r_task = r_vel - 0.02 * m_still * sum(|qdot|)",
    task_terms=(_base_velocity_term(), _standstill_joint_vel_penalty_term()),
  ),
  Exp9RewardSpec(
    key="B",
    name="joint_vel_gate",
    summary="B: r_task = r_vel * ((1-m_still) + m_still*r_joint_vel)",
    task_terms=(
      (
        mdp.forward_stop_joint_vel_gate_task,
        1.0,
        {
          "command_name": "steering",
          "vel_err_scale": _VEL_ERR_SCALE,
          "zero_threshold": _ZERO_THRESHOLD,
          "joint_vel_exp_scale": _JOINT_VEL_EXP_SCALE,
        },
      ),
    ),
  ),
  Exp9RewardSpec(
    key="C",
    name="stop_sum_switch",
    summary="C: moving uses r_vel; stopping uses 0.6*r_root_stop + 0.4*r_joint_vel",
    task_terms=(
      (
        mdp.forward_stop_switch_sum_task,
        1.0,
        {
          "command_name": "steering",
          "vel_err_scale": _VEL_ERR_SCALE,
          "zero_threshold": _ZERO_THRESHOLD,
          "root_vel_exp_scale": _ROOT_VEL_EXP_SCALE,
          "joint_vel_exp_scale": _JOINT_VEL_EXP_SCALE,
          "root_stop_weight": 0.6,
          "joint_vel_weight": 0.4,
        },
      ),
    ),
  ),
  Exp9RewardSpec(
    key="D",
    name="stop_product_switch",
    summary="D: moving uses r_vel; stopping uses r_root_stop * r_joint_vel",
    task_terms=(
      (
        mdp.forward_stop_switch_product_task,
        1.0,
        {
          "command_name": "steering",
          "vel_err_scale": _VEL_ERR_SCALE,
          "zero_threshold": _ZERO_THRESHOLD,
          "root_vel_exp_scale": _ROOT_VEL_EXP_SCALE,
          "joint_vel_exp_scale": _JOINT_VEL_EXP_SCALE,
        },
      ),
    ),
  ),
  Exp9RewardSpec(
    key="E",
    name="stop_product_mix_switch",
    summary=(
      "E: moving uses r_vel; stopping uses "
      "0.6*r_root_stop*r_joint_vel + 0.2*r_root_stop + 0.2*r_joint_vel"
    ),
    task_terms=(
      (
        mdp.forward_stop_switch_mix_product_task,
        1.0,
        {
          "command_name": "steering",
          "vel_err_scale": _VEL_ERR_SCALE,
          "zero_threshold": _ZERO_THRESHOLD,
          "root_vel_exp_scale": _ROOT_VEL_EXP_SCALE,
          "joint_vel_exp_scale": _JOINT_VEL_EXP_SCALE,
          "product_weight": 0.6,
          "root_stop_weight": 0.2,
          "joint_vel_weight": 0.2,
        },
      ),
    ),
  ),
)

EXP9_REWARD_ROOT_ANG = Exp9RewardSpec(
  key="F",
  name="stop_product_root_ang",
  summary=(
    "F: moving uses r_vel; stopping uses "
    "r_root_stop * r_joint_vel * r_root_ang"
  ),
  task_terms=(
    (
      mdp.forward_stop_product_root_ang_task,
      1.0,
      {
        "command_name": "steering",
        "vel_err_scale": _VEL_ERR_SCALE,
        "zero_threshold": _ZERO_THRESHOLD,
        "root_vel_exp_scale": _ROOT_VEL_EXP_SCALE,
        "joint_vel_exp_scale": _JOINT_VEL_EXP_SCALE,
        "root_ang_exp_scale": _ROOT_ANG_EXP_SCALE,
      },
    ),
  ),
)

EXP9_REWARD_UPPER_BODY = Exp9RewardSpec(
  key="G",
  name="stop_product_upper_body",
  summary=(
    "G: moving uses r_vel; stopping uses "
    "r_root_stop * r_joint_vel * r_upper_joint_vel"
  ),
  task_terms=(
    (
      mdp.forward_stop_product_upper_body_joint_task,
      1.0,
      {
        "command_name": "steering",
        "vel_err_scale": _VEL_ERR_SCALE,
        "zero_threshold": _ZERO_THRESHOLD,
        "root_vel_exp_scale": _ROOT_VEL_EXP_SCALE,
        "joint_vel_exp_scale": _JOINT_VEL_EXP_SCALE,
        "upper_joint_vel_exp_scale": _UPPER_JOINT_VEL_EXP_SCALE,
      },
    ),
  ),
)

EXP9_REWARD_ACTION_SMOOTH = Exp9RewardSpec(
  key="H",
  name="stop_product_action_smooth",
  summary=(
    "H: moving uses r_vel; stopping uses "
    "r_root_stop * r_joint_vel * r_action_smooth"
  ),
  task_terms=(
    (
      mdp.forward_stop_product_action_smooth_task,
      1.0,
      {
        "command_name": "steering",
        "vel_err_scale": _VEL_ERR_SCALE,
        "zero_threshold": _ZERO_THRESHOLD,
        "root_vel_exp_scale": _ROOT_VEL_EXP_SCALE,
        "joint_vel_exp_scale": _JOINT_VEL_EXP_SCALE,
        "action_smooth_exp_scale": _ACTION_SMOOTH_EXP_SCALE,
      },
    ),
  ),
)


_EXP9_BASE_GROUP_SPECS: tuple[Exp9GroupSpec, ...] = tuple(
  Exp9GroupSpec(
    group=(prior_idx * len(EXP9_REWARDS) + reward_idx + 1),
    prior=prior,
    reward=reward,
    run_name=(
      f"group{prior_idx * len(EXP9_REWARDS) + reward_idx + 1:02d}_"
      f"p{prior.index}_{reward.key.lower()}_{reward.name}"
    ),
  )
  for prior_idx, prior in enumerate(EXP9_PRIORS)
  for reward_idx, reward in enumerate(EXP9_REWARDS)
)

_P4 = EXP9_PRIORS[3]
_REWARD_D = EXP9_REWARDS[3]
_EXP9_EXTRA_GROUP_SPECS: tuple[Exp9GroupSpec, ...] = (
  Exp9GroupSpec(21, _P4, EXP9_REWARD_ROOT_ANG, "group21_p4_f_root_ang_stop"),
  Exp9GroupSpec(22, _P4, EXP9_REWARD_UPPER_BODY, "group22_p4_g_upper_body_stop"),
  Exp9GroupSpec(23, _P4, EXP9_REWARD_ACTION_SMOOTH, "group23_p4_h_action_smooth_stop"),
  Exp9GroupSpec(
    24,
    _P4,
    EXP9_REWARD_ROOT_ANG,
    "group24_p4_f_root_ang_stop_dz1",
    dead_zone_speed=_DEAD_ZONE_SPEED,
  ),
  Exp9GroupSpec(
    25,
    _P4,
    EXP9_REWARD_UPPER_BODY,
    "group25_p4_g_upper_body_stop_dz1",
    dead_zone_speed=_DEAD_ZONE_SPEED,
  ),
  Exp9GroupSpec(
    26,
    _P4,
    EXP9_REWARD_ACTION_SMOOTH,
    "group26_p4_h_action_smooth_stop_dz1",
    dead_zone_speed=_DEAD_ZONE_SPEED,
  ),
  Exp9GroupSpec(27, EXP9_PRIOR5, EXP9_REWARD_ROOT_ANG, "group27_p5_f_root_ang_stop"),
  Exp9GroupSpec(28, EXP9_PRIOR5, EXP9_REWARD_UPPER_BODY, "group28_p5_g_upper_body_stop"),
  Exp9GroupSpec(
    29,
    EXP9_PRIOR5,
    EXP9_REWARD_ACTION_SMOOTH,
    "group29_p5_h_action_smooth_stop",
  ),
  Exp9GroupSpec(
    30,
    EXP9_PRIOR5,
    EXP9_REWARD_ROOT_ANG,
    "group30_p5_f_root_ang_stop_dz1",
    dead_zone_speed=_DEAD_ZONE_SPEED,
  ),
  Exp9GroupSpec(
    31,
    EXP9_PRIOR5,
    EXP9_REWARD_UPPER_BODY,
    "group31_p5_g_upper_body_stop_dz1",
    dead_zone_speed=_DEAD_ZONE_SPEED,
  ),
  Exp9GroupSpec(
    32,
    EXP9_PRIOR5,
    EXP9_REWARD_ACTION_SMOOTH,
    "group32_p5_h_action_smooth_stop_dz1",
    dead_zone_speed=_DEAD_ZONE_SPEED,
  ),
  Exp9GroupSpec(
    33,
    EXP9_PRIOR5,
    _REWARD_D,
    "group33_p5_d_stop_product_switch_baseline",
  ),
)

EXP9_GROUP_SPECS: tuple[Exp9GroupSpec, ...] = (
  _EXP9_BASE_GROUP_SPECS + _EXP9_EXTRA_GROUP_SPECS
)


def _build_exp9_cfg(spec: Exp9GroupSpec, play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = g1_forward_exp6_group7_smp_env_cfg(play=play)
  cfg.commands["steering"].dead_zone_speed = spec.dead_zone_speed
  cfg.events["init_smp_state"].params["ckpt_path"] = spec.prior.ckpt_path
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_product,
    weight=1.0,
    params={"task_terms": spec.reward.task_terms},
  )
  return cfg


def _make_builder(spec: Exp9GroupSpec) -> Callable[[bool], ManagerBasedRlEnvCfg]:
  def _builder(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _build_exp9_cfg(spec, play=play)

  _builder.__name__ = f"g1_forward_exp9_group{spec.group}_smp_env_cfg"
  return _builder


EXP9_FORWARD_BUILDERS = tuple(
  (spec.group, _make_builder(spec), spec) for spec in EXP9_GROUP_SPECS
)
