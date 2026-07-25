"""Body-velocity task reward components."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from smp.rl.tasks.body_velocity.mdp.commands import BodyVelocityCommand


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_FEET_ASSET_CFG = SceneEntityCfg(
  "robot", body_names=("left_ankle_roll_link", "right_ankle_roll_link")
)


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


def _body_link_quat_w(data) -> torch.Tensor:
  if hasattr(data, "body_link_quat_w"):
    return data.body_link_quat_w
  return data.body_quat_w


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
  product_weight: float,
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
  return product_weight * linear * yaw + linear_weight * linear + yaw_weight * yaw


def _command_norm(
  env: "ManagerBasedRlEnv",
  command_name: str,
) -> torch.Tensor:
  cmd: "BodyVelocityCommand" = env.command_manager.get_term(command_name)  # type: ignore[assignment]
  command = torch.cat([cmd.lin_vel_b, cmd.yaw_rate.unsqueeze(-1)], dim=-1)
  return torch.linalg.norm(command, dim=-1)


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
  moving_product_weight: float = 0.0,
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
    moving_product_weight,
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
  moving_product_weight: float = 0.0,
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
    moving_product_weight,
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
  moving_product_weight: float = 0.0,
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
    moving_product_weight,
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


def support_foot_tilt_penalty(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  contact_threshold: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_FEET_ASSET_CFG,
) -> torch.Tensor:
  """Penalty for tilted support feet.

  The term is active only for feet whose contact force exceeds
  ``contact_threshold``. A flat support foot has its local z-axis close to world
  z, so the xy components of that up vector are near zero.
  """
  contact_sensor = env.scene.sensors[sensor_name]
  force = contact_sensor.data.force
  if force is None:
    msg = f"Contact sensor '{sensor_name}' must include force fields."
    raise RuntimeError(msg)

  in_contact = torch.linalg.norm(force, dim=-1) > contact_threshold
  if in_contact.ndim > 2:
    in_contact = in_contact.any(dim=tuple(range(2, in_contact.ndim)))

  asset = env.scene[asset_cfg.name]
  body_quat_w = _body_link_quat_w(asset.data)[:, asset_cfg.body_ids]
  local_up = torch.zeros((*body_quat_w.shape[:-1], 3), device=body_quat_w.device)
  local_up[..., 2] = 1.0
  foot_up_w = quat_apply(body_quat_w.reshape(-1, 4), local_up.reshape(-1, 3)).reshape_as(
    local_up
  )
  tilt = torch.sum(torch.square(foot_up_w[..., :2]), dim=-1)
  return torch.sum(tilt * in_contact.to(tilt.dtype), dim=1)


def static_double_support_force_penalty(
  env: "ManagerBasedRlEnv",
  command_name: str,
  sensor_name: str,
  min_contact_force: float = 80.0,
  command_threshold: float = 0.2,
) -> torch.Tensor:
  """Count under-loaded feet while a zero body-velocity command is active.

  Each foot is tested independently against the magnitude of its terrain net
  contact force. A zero-command environment receives one penalty unit for
  each foot below ``min_contact_force``.
  """
  contact_sensor = env.scene.sensors[sensor_name]
  force = contact_sensor.data.force
  if force is None:
    msg = f"Contact sensor '{sensor_name}' must include force fields."
    raise RuntimeError(msg)

  force_norm = torch.linalg.norm(force, dim=-1)
  if force_norm.ndim > 2:
    force_norm = torch.amax(force_norm, dim=tuple(range(2, force_norm.ndim)))
  under_loaded = force_norm < min_contact_force
  still = _command_norm(env, command_name) < command_threshold
  return torch.sum(under_loaded.to(torch.float32), dim=1) * still.to(torch.float32)


def persistent_single_support_penalty(
  env: "ManagerBasedRlEnv",
  command_name: str,
  sensor_name: str,
  max_single_support_time: float = 0.45,
  max_excess_time: float = 0.8,
  command_threshold: float = 0.2,
) -> torch.Tensor:
  """Penalty for overlong one-foot support while a body-velocity command is active."""
  contact_sensor = env.scene.sensors[sensor_name]
  contact_time = contact_sensor.data.current_contact_time
  if contact_time is None:
    msg = f"Contact sensor '{sensor_name}' must enable track_air_time."
    raise RuntimeError(msg)

  in_contact = contact_time > 0.0
  single_stance = torch.sum(in_contact, dim=1) == 1
  support_time = torch.max(contact_time, dim=1).values
  excess = (support_time - max_single_support_time).clamp(
    min=0.0, max=max_excess_time
  )
  moving = _command_norm(env, command_name) > command_threshold
  return excess * single_stance.to(excess.dtype) * moving.to(excess.dtype)


def double_air_penalty(
  env: "ManagerBasedRlEnv",
  command_name: str,
  sensor_name: str,
  command_threshold: float = 0.2,
) -> torch.Tensor:
  """Penalty when both feet are airborne while a body-velocity command is active."""
  contact_sensor = env.scene.sensors[sensor_name]
  contact_time = contact_sensor.data.current_contact_time
  if contact_time is None:
    msg = f"Contact sensor '{sensor_name}' must enable track_air_time."
    raise RuntimeError(msg)

  in_contact = contact_time > 0.0
  no_support = torch.sum(in_contact, dim=1) == 0
  moving = _command_norm(env, command_name) > command_threshold
  return no_support.to(torch.float32) * moving.to(torch.float32)
