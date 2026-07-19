"""Body-velocity task reward components."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from smp.rl.tasks.body_velocity.mdp.commands import BodyVelocityCommand


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _root_lin_vel_w(data) -> torch.Tensor:
  return data.root_link_lin_vel_w if hasattr(data, "root_link_lin_vel_w") else data.root_lin_vel_w


def _root_ang_vel_w(data) -> torch.Tensor:
  return data.root_link_ang_vel_w if hasattr(data, "root_link_ang_vel_w") else data.root_ang_vel_w


def _root_lin_vel_b(data) -> torch.Tensor:
  if hasattr(data, "root_link_lin_vel_b"):
    return data.root_link_lin_vel_b
  if hasattr(data, "root_lin_vel_b"):
    return data.root_lin_vel_b
  root_lin_vel_w = _root_lin_vel_w(data)
  heading_w = data.heading_w
  cos_h = torch.cos(heading_w)
  sin_h = torch.sin(heading_w)
  x_w, y_w = root_lin_vel_w[:, 0], root_lin_vel_w[:, 1]
  xy_b = torch.stack([cos_h * x_w + sin_h * y_w, -sin_h * x_w + cos_h * y_w], dim=-1)
  return torch.cat([xy_b, root_lin_vel_w[:, 2:3]], dim=-1)


def _root_yaw_rate(data) -> torch.Tensor:
  if hasattr(data, "root_link_ang_vel_b"):
    return data.root_link_ang_vel_b[:, 2]
  if hasattr(data, "root_ang_vel_b"):
    return data.root_ang_vel_b[:, 2]
  return _root_ang_vel_w(data)[:, 2]


def _standstill_mask(
  env: "ManagerBasedRlEnv",
  command_name: str,
  command_zero_threshold: float,
) -> torch.Tensor:
  cmd: "BodyVelocityCommand" = env.command_manager.get_term(command_name)  # type: ignore[assignment]
  command = torch.cat([cmd.lin_vel_b, cmd.yaw_rate.unsqueeze(-1)], dim=-1)
  return torch.linalg.norm(command, dim=-1) < command_zero_threshold


def _body_velocity_task(
  env: "ManagerBasedRlEnv",
  command_name: str,
  lin_vel_err_scale: float,
  yaw_rate_err_scale: float,
  linear_weight: float,
  yaw_weight: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  linear = body_velocity_linear_tracking(
    env,
    command_name=command_name,
    lin_vel_err_scale=lin_vel_err_scale,
    asset_cfg=asset_cfg,
  )
  yaw = body_velocity_yaw_tracking(
    env,
    command_name=command_name,
    yaw_rate_err_scale=yaw_rate_err_scale,
    asset_cfg=asset_cfg,
  )
  return linear_weight * linear + yaw_weight * yaw


def body_velocity_linear_tracking(
  env: "ManagerBasedRlEnv",
  command_name: str,
  lin_vel_err_scale: float = 2.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """``exp(-lin_vel_err_scale * ||v_xy_body - v_xy_cmd_body||^2)``."""
  asset = env.scene[asset_cfg.name]
  cmd: "BodyVelocityCommand" = env.command_manager.get_term(command_name)  # type: ignore[assignment]

  root_lin_vel_b = _root_lin_vel_b(asset.data)
  lin_vel_err = torch.sum((root_lin_vel_b[:, :2] - cmd.lin_vel_b) ** 2, dim=-1)
  return torch.exp(-lin_vel_err_scale * lin_vel_err)


def body_velocity_linear_tracking_zero_negative_projection(
  env: "ManagerBasedRlEnv",
  command_name: str,
  lin_vel_err_scale: float = 2.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Body-velocity linear tracking, zeroed when root velocity opposes command."""
  asset = env.scene[asset_cfg.name]
  cmd: "BodyVelocityCommand" = env.command_manager.get_term(command_name)  # type: ignore[assignment]

  reward = body_velocity_linear_tracking(
    env,
    command_name=command_name,
    lin_vel_err_scale=lin_vel_err_scale,
    asset_cfg=asset_cfg,
  )
  root_lin_vel_b = _root_lin_vel_b(asset.data)
  projection = torch.sum(root_lin_vel_b[:, :2] * cmd.lin_vel_b, dim=-1)
  return torch.where(projection < 0.0, torch.zeros_like(reward), reward)


def body_velocity_yaw_tracking(
  env: "ManagerBasedRlEnv",
  command_name: str,
  yaw_rate_err_scale: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """``exp(-yaw_rate_err_scale * (yaw_rate_body - yaw_rate_cmd)^2)``."""
  asset = env.scene[asset_cfg.name]
  cmd: "BodyVelocityCommand" = env.command_manager.get_term(command_name)  # type: ignore[assignment]

  yaw_rate_err = (cmd.yaw_rate - _root_yaw_rate(asset.data)) ** 2
  return torch.exp(-yaw_rate_err_scale * yaw_rate_err)


def body_velocity_stop_switch_sum_task(
  env: "ManagerBasedRlEnv",
  command_name: str,
  lin_vel_err_scale: float = 2.0,
  yaw_rate_err_scale: float = 1.0,
  command_zero_threshold: float = 0.2,
  root_vel_exp_scale: float = 2.0,
  joint_vel_exp_scale: float = 0.02,
  moving_linear_weight: float = 0.75,
  moving_yaw_weight: float = 0.25,
  root_stop_weight: float = 0.6,
  joint_vel_weight: float = 0.4,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Exp10 body-velocity reward C: original body-velocity task when moving, stop-sum when still."""
  asset = env.scene[asset_cfg.name]
  still = _standstill_mask(env, command_name, command_zero_threshold).to(
    asset.data.joint_vel.dtype
  )
  r_move = _body_velocity_task(
    env,
    command_name,
    lin_vel_err_scale,
    yaw_rate_err_scale,
    moving_linear_weight,
    moving_yaw_weight,
    asset_cfg,
  )
  r_root_stop = torch.exp(
    -root_vel_exp_scale * (_root_lin_vel_w(asset.data)[:, :2] ** 2).sum(dim=-1)
  )
  r_joint_vel = torch.exp(
    -joint_vel_exp_scale * torch.abs(asset.data.joint_vel).sum(dim=-1)
  )
  r_stop = root_stop_weight * r_root_stop + joint_vel_weight * r_joint_vel
  return (1.0 - still) * r_move + still * r_stop


def body_velocity_stop_switch_product_task(
  env: "ManagerBasedRlEnv",
  command_name: str,
  lin_vel_err_scale: float = 2.0,
  yaw_rate_err_scale: float = 1.0,
  command_zero_threshold: float = 0.2,
  root_vel_exp_scale: float = 2.0,
  joint_vel_exp_scale: float = 0.02,
  moving_linear_weight: float = 0.75,
  moving_yaw_weight: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Exp10 body-velocity reward D: original body-velocity task when moving, stop-product when still."""
  asset = env.scene[asset_cfg.name]
  still = _standstill_mask(env, command_name, command_zero_threshold).to(
    asset.data.joint_vel.dtype
  )
  r_move = _body_velocity_task(
    env,
    command_name,
    lin_vel_err_scale,
    yaw_rate_err_scale,
    moving_linear_weight,
    moving_yaw_weight,
    asset_cfg,
  )
  r_root_stop = torch.exp(
    -root_vel_exp_scale * (_root_lin_vel_w(asset.data)[:, :2] ** 2).sum(dim=-1)
  )
  r_joint_vel = torch.exp(
    -joint_vel_exp_scale * torch.abs(asset.data.joint_vel).sum(dim=-1)
  )
  return (1.0 - still) * r_move + still * (r_root_stop * r_joint_vel)


def body_velocity_stop_switch_mix_product_task(
  env: "ManagerBasedRlEnv",
  command_name: str,
  lin_vel_err_scale: float = 2.0,
  yaw_rate_err_scale: float = 1.0,
  command_zero_threshold: float = 0.2,
  root_vel_exp_scale: float = 2.0,
  joint_vel_exp_scale: float = 0.02,
  moving_linear_weight: float = 0.75,
  moving_yaw_weight: float = 0.25,
  product_weight: float = 0.6,
  root_stop_weight: float = 0.2,
  joint_vel_weight: float = 0.2,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Exp10 body-velocity reward E: original body-velocity task when moving, mixed stop-product when still."""
  asset = env.scene[asset_cfg.name]
  still = _standstill_mask(env, command_name, command_zero_threshold).to(
    asset.data.joint_vel.dtype
  )
  r_move = _body_velocity_task(
    env,
    command_name,
    lin_vel_err_scale,
    yaw_rate_err_scale,
    moving_linear_weight,
    moving_yaw_weight,
    asset_cfg,
  )
  r_root_stop = torch.exp(
    -root_vel_exp_scale * (_root_lin_vel_w(asset.data)[:, :2] ** 2).sum(dim=-1)
  )
  r_joint_vel = torch.exp(
    -joint_vel_exp_scale * torch.abs(asset.data.joint_vel).sum(dim=-1)
  )
  r_stop = (
    product_weight * r_root_stop * r_joint_vel
    + root_stop_weight * r_root_stop
    + joint_vel_weight * r_joint_vel
  )
  return (1.0 - still) * r_move + still * r_stop
