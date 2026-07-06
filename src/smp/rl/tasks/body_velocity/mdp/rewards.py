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
