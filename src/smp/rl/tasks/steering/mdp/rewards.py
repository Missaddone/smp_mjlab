"""Steering reward components: linear-velocity tracking + face alignment.

SMP-gated via the generic ``smp.rl.rewards.smp_product``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from smp.rl.tasks.steering.mdp.commands import SteeringCommand


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_FEET_ASSET_CFG = SceneEntityCfg(
  "robot", body_names=("left_ankle_roll_link", "right_ankle_roll_link")
)
_DEFAULT_UPPER_BODY_ASSET_CFG = SceneEntityCfg(
  "robot",
  joint_names=(
    r"waist_.*_joint",
    r".*shoulder.*_joint",
    r".*elbow.*_joint",
    r".*wrist.*_joint",
  ),
)


def steering_target_velocity(
  env: "ManagerBasedRlEnv",
  command_name: str,
  vel_err_scale: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """``exp(-vel_err_scale * ‖tar_speed·tar_dir - root_vel_xy‖²)``, zeroed when
  root velocity projects negatively onto the target dir (no reward for walking
  the wrong way)."""
  asset = env.scene[asset_cfg.name]
  cmd: "SteeringCommand" = env.command_manager.get_term(command_name)  # type: ignore[assignment]

  root_vel_xy = asset.data.root_link_lin_vel_w[:, :2]
  tar_vel = cmd.tar_speed.unsqueeze(-1) * cmd.tar_dir_w
  vel_err = ((tar_vel - root_vel_xy) ** 2).sum(dim=-1)

  proj_speed = (cmd.tar_dir_w * root_vel_xy).sum(dim=-1)
  reward = torch.exp(-vel_err_scale * vel_err)
  reward = torch.where(proj_speed < 0, torch.zeros_like(reward), reward)
  return reward


def steering_face_direction(
  env: "ManagerBasedRlEnv",
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """``max(face_dir · char_face_dir, 0)`` — both unit world-xy vectors."""
  asset = env.scene[asset_cfg.name]
  cmd: "SteeringCommand" = env.command_manager.get_term(command_name)  # type: ignore[assignment]

  heading_w = asset.data.heading_w
  char_face_w = torch.stack([torch.cos(heading_w), torch.sin(heading_w)], dim=-1)
  face_dot = (cmd.face_dir_w * char_face_w).sum(dim=-1)
  return face_dot.clamp_min(0.0)


def _standstill_mask(
  env: "ManagerBasedRlEnv",
  command_name: str,
  zero_threshold: float,
) -> torch.Tensor:
  cmd: "SteeringCommand" = env.command_manager.get_term(command_name)  # type: ignore[assignment]
  tar_vel = cmd.tar_speed.unsqueeze(-1) * cmd.tar_dir_w
  return torch.linalg.norm(tar_vel, dim=-1) <= zero_threshold


def stand_still_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  default_joint_pos: Sequence[float],
  zero_threshold: float = 0.2,
  joint_pos_scale: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """``exp(-scale * sum(abs(q - q_default)))`` when the speed command is zero."""
  asset = env.scene[asset_cfg.name]
  default = torch.tensor(
    default_joint_pos, device=asset.data.joint_pos.device, dtype=asset.data.joint_pos.dtype
  )
  pose_err = torch.abs(asset.data.joint_pos - default.unsqueeze(0)).sum(dim=-1)
  reward = torch.exp(-joint_pos_scale * pose_err)
  return reward * _standstill_mask(env, command_name, zero_threshold).to(reward.dtype)


def stand_still_vel(
  env: "ManagerBasedRlEnv",
  command_name: str,
  zero_threshold: float = 0.2,
  joint_vel_scale: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Joint-velocity penalty magnitude, active only for zero-speed commands."""
  asset = env.scene[asset_cfg.name]
  penalty = joint_vel_scale * torch.abs(asset.data.joint_vel).sum(dim=-1)
  return penalty * _standstill_mask(env, command_name, zero_threshold).to(penalty.dtype)


def stand_still_feet_motion_penalty(
  env: "ManagerBasedRlEnv",
  command_name: str,
  zero_threshold: float = 0.2,
  foot_vel_scale: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_FEET_ASSET_CFG,
) -> torch.Tensor:
  """Foot linear-velocity penalty magnitude, active only for zero-speed commands."""
  asset = env.scene[asset_cfg.name]
  foot_vel = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]
  penalty = foot_vel_scale * torch.linalg.norm(foot_vel, dim=-1).sum(dim=-1)
  return penalty * _standstill_mask(env, command_name, zero_threshold).to(penalty.dtype)


def stand_still_double_support(
  env: "ManagerBasedRlEnv",
  command_name: str,
  sensor_name: str,
  zero_threshold: float = 0.2,
  force_threshold: float = 2.0,
) -> torch.Tensor:
  """Reward both left and right foot contact when the speed command is zero."""
  sensor = env.scene[sensor_name]
  data = sensor.data
  primary_names = sensor.primary_names

  if data.force is not None:
    contact_values = data.force[..., 2] > force_threshold
  elif data.found is not None:
    contact_values = data.found > 0
  else:
    raise RuntimeError(
      f"Contact sensor '{sensor_name}' must expose either force or found data."
    )

  if contact_values.ndim == 3:
    contact_values = contact_values.any(dim=-1)

  left_ids = [
    idx for idx, name in enumerate(primary_names) if name.startswith("left_foot")
  ]
  right_ids = [
    idx for idx, name in enumerate(primary_names) if name.startswith("right_foot")
  ]
  if not left_ids or not right_ids:
    raise RuntimeError(
      f"Contact sensor '{sensor_name}' did not resolve left/right foot primaries: "
      f"{primary_names}"
    )

  left_contact = contact_values[:, left_ids].any(dim=-1)
  right_contact = contact_values[:, right_ids].any(dim=-1)
  reward = (left_contact & right_contact).to(dtype=torch.float32)
  return reward * _standstill_mask(env, command_name, zero_threshold).to(reward.dtype)


def forward_stop_joint_vel_gate_task(
  env: "ManagerBasedRlEnv",
  command_name: str,
  vel_err_scale: float = 0.5,
  zero_threshold: float = 0.2,
  joint_vel_exp_scale: float = 0.02,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Exp9 reward B: keep velocity tracking outside stop, gate it by joint stillness
  when the command asks for stop."""
  asset = env.scene[asset_cfg.name]
  still = _standstill_mask(env, command_name, zero_threshold).to(asset.data.joint_vel.dtype)
  r_vel = steering_target_velocity(
    env,
    command_name=command_name,
    vel_err_scale=vel_err_scale,
    asset_cfg=asset_cfg,
  )
  r_joint_vel = torch.exp(
    -joint_vel_exp_scale * torch.abs(asset.data.joint_vel).sum(dim=-1)
  )
  return r_vel * ((1.0 - still) + still * r_joint_vel)


def forward_stop_switch_sum_task(
  env: "ManagerBasedRlEnv",
  command_name: str,
  vel_err_scale: float = 0.5,
  zero_threshold: float = 0.2,
  root_vel_exp_scale: float = 2.0,
  joint_vel_exp_scale: float = 0.02,
  root_stop_weight: float = 0.6,
  joint_vel_weight: float = 0.4,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Exp9 reward C: normal tracking when moving; weighted root-stop/joint-still
  reward when stopping."""
  asset = env.scene[asset_cfg.name]
  still = _standstill_mask(env, command_name, zero_threshold).to(asset.data.joint_vel.dtype)
  r_vel = steering_target_velocity(
    env,
    command_name=command_name,
    vel_err_scale=vel_err_scale,
    asset_cfg=asset_cfg,
  )
  r_root_stop = torch.exp(
    -root_vel_exp_scale * (asset.data.root_link_lin_vel_w[:, :2] ** 2).sum(dim=-1)
  )
  r_joint_vel = torch.exp(
    -joint_vel_exp_scale * torch.abs(asset.data.joint_vel).sum(dim=-1)
  )
  r_stop = root_stop_weight * r_root_stop + joint_vel_weight * r_joint_vel
  return (1.0 - still) * r_vel + still * r_stop


def forward_stop_switch_product_task(
  env: "ManagerBasedRlEnv",
  command_name: str,
  vel_err_scale: float = 0.5,
  zero_threshold: float = 0.2,
  root_vel_exp_scale: float = 2.0,
  joint_vel_exp_scale: float = 0.02,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Exp9 reward D: normal tracking when moving; product root-stop and joint-still
  reward when stopping."""
  asset = env.scene[asset_cfg.name]
  still = _standstill_mask(env, command_name, zero_threshold).to(asset.data.joint_vel.dtype)
  r_vel = steering_target_velocity(
    env,
    command_name=command_name,
    vel_err_scale=vel_err_scale,
    asset_cfg=asset_cfg,
  )
  r_root_stop = torch.exp(
    -root_vel_exp_scale * (asset.data.root_link_lin_vel_w[:, :2] ** 2).sum(dim=-1)
  )
  r_joint_vel = torch.exp(
    -joint_vel_exp_scale * torch.abs(asset.data.joint_vel).sum(dim=-1)
  )
  return (1.0 - still) * r_vel + still * (r_root_stop * r_joint_vel)


def forward_stop_switch_mix_product_task(
  env: "ManagerBasedRlEnv",
  command_name: str,
  vel_err_scale: float = 0.5,
  zero_threshold: float = 0.2,
  root_vel_exp_scale: float = 2.0,
  joint_vel_exp_scale: float = 0.02,
  product_weight: float = 0.6,
  root_stop_weight: float = 0.2,
  joint_vel_weight: float = 0.2,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Exp9 reward E: normal tracking when moving; product-plus-marginals stop
  reward when stopping."""
  asset = env.scene[asset_cfg.name]
  still = _standstill_mask(env, command_name, zero_threshold).to(asset.data.joint_vel.dtype)
  r_vel = steering_target_velocity(
    env,
    command_name=command_name,
    vel_err_scale=vel_err_scale,
    asset_cfg=asset_cfg,
  )
  r_root_stop = torch.exp(
    -root_vel_exp_scale * (asset.data.root_link_lin_vel_w[:, :2] ** 2).sum(dim=-1)
  )
  r_joint_vel = torch.exp(
    -joint_vel_exp_scale * torch.abs(asset.data.joint_vel).sum(dim=-1)
  )
  r_stop = (
    product_weight * r_root_stop * r_joint_vel
    + root_stop_weight * r_root_stop
    + joint_vel_weight * r_joint_vel
  )
  return (1.0 - still) * r_vel + still * r_stop


def forward_stop_product_root_ang_task(
  env: "ManagerBasedRlEnv",
  command_name: str,
  vel_err_scale: float = 0.5,
  zero_threshold: float = 0.2,
  root_vel_exp_scale: float = 2.0,
  joint_vel_exp_scale: float = 0.02,
  root_ang_exp_scale: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Exp9 extension: group19 stop product plus root roll/pitch angular stillness."""
  asset = env.scene[asset_cfg.name]
  still = _standstill_mask(env, command_name, zero_threshold).to(asset.data.joint_vel.dtype)
  r_vel = steering_target_velocity(
    env,
    command_name=command_name,
    vel_err_scale=vel_err_scale,
    asset_cfg=asset_cfg,
  )
  r_root_stop = torch.exp(
    -root_vel_exp_scale * (asset.data.root_link_lin_vel_w[:, :2] ** 2).sum(dim=-1)
  )
  r_joint_vel = torch.exp(
    -joint_vel_exp_scale * torch.abs(asset.data.joint_vel).sum(dim=-1)
  )
  r_root_ang = torch.exp(
    -root_ang_exp_scale * (asset.data.root_link_ang_vel_w[:, :2] ** 2).sum(dim=-1)
  )
  return (1.0 - still) * r_vel + still * (r_root_stop * r_joint_vel * r_root_ang)


def forward_stop_product_upper_body_joint_task(
  env: "ManagerBasedRlEnv",
  command_name: str,
  vel_err_scale: float = 0.5,
  zero_threshold: float = 0.2,
  root_vel_exp_scale: float = 2.0,
  joint_vel_exp_scale: float = 0.02,
  upper_joint_vel_exp_scale: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  upper_asset_cfg: SceneEntityCfg = _DEFAULT_UPPER_BODY_ASSET_CFG,
) -> torch.Tensor:
  """Exp9 extension: group19 stop product plus upper-body joint stillness."""
  asset = env.scene[asset_cfg.name]
  upper_asset = env.scene[upper_asset_cfg.name]
  still = _standstill_mask(env, command_name, zero_threshold).to(asset.data.joint_vel.dtype)
  r_vel = steering_target_velocity(
    env,
    command_name=command_name,
    vel_err_scale=vel_err_scale,
    asset_cfg=asset_cfg,
  )
  r_root_stop = torch.exp(
    -root_vel_exp_scale * (asset.data.root_link_lin_vel_w[:, :2] ** 2).sum(dim=-1)
  )
  r_joint_vel = torch.exp(
    -joint_vel_exp_scale * torch.abs(asset.data.joint_vel).sum(dim=-1)
  )
  upper_joint_vel = upper_asset.data.joint_vel[:, upper_asset_cfg.joint_ids]
  r_upper_joint_vel = torch.exp(
    -upper_joint_vel_exp_scale * torch.abs(upper_joint_vel).sum(dim=-1)
  )
  return (
    (1.0 - still) * r_vel
    + still * (r_root_stop * r_joint_vel * r_upper_joint_vel)
  )


def forward_stop_product_action_smooth_task(
  env: "ManagerBasedRlEnv",
  command_name: str,
  vel_err_scale: float = 0.5,
  zero_threshold: float = 0.2,
  root_vel_exp_scale: float = 2.0,
  joint_vel_exp_scale: float = 0.02,
  action_smooth_exp_scale: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Exp9 extension: group19 stop product plus raw-policy action smoothness."""
  asset = env.scene[asset_cfg.name]
  still = _standstill_mask(env, command_name, zero_threshold).to(asset.data.joint_vel.dtype)
  r_vel = steering_target_velocity(
    env,
    command_name=command_name,
    vel_err_scale=vel_err_scale,
    asset_cfg=asset_cfg,
  )
  r_root_stop = torch.exp(
    -root_vel_exp_scale * (asset.data.root_link_lin_vel_w[:, :2] ** 2).sum(dim=-1)
  )
  r_joint_vel = torch.exp(
    -joint_vel_exp_scale * torch.abs(asset.data.joint_vel).sum(dim=-1)
  )
  action_delta = env.action_manager.action - env.action_manager.prev_action
  r_action_smooth = torch.exp(
    -action_smooth_exp_scale * (action_delta**2).sum(dim=-1)
  )
  return (
    (1.0 - still) * r_vel
    + still * (r_root_stop * r_joint_vel * r_action_smooth)
  )
