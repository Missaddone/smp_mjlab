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

  tilt = _per_foot_tilt(env, asset_cfg)
  return torch.sum(tilt * in_contact.to(tilt.dtype), dim=1)


def _per_foot_tilt(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Return raw ``sin^2(theta)`` tilt for each configured foot body."""
  asset = env.scene[asset_cfg.name]
  body_quat_w = _body_link_quat_w(asset.data)[:, asset_cfg.body_ids]
  local_up = torch.zeros((*body_quat_w.shape[:-1], 3), device=body_quat_w.device)
  local_up[..., 2] = 1.0
  foot_up_w = quat_apply(body_quat_w.reshape(-1, 4), local_up.reshape(-1, 3)).reshape_as(
    local_up
  )
  return torch.sum(torch.square(foot_up_w[..., :2]), dim=-1)


def static_foot_tilt_penalty(
  env: "ManagerBasedRlEnv",
  command_name: str,
  foot_index: int,
  command_zero_threshold: float = 0.2,
  asset_cfg: SceneEntityCfg = _DEFAULT_FEET_ASSET_CFG,
) -> torch.Tensor:
  """Return one foot's raw tilt only while a static command is active."""
  if not 0 <= foot_index < len(asset_cfg.body_ids):
    raise ValueError(f"foot_index must be in 0..{len(asset_cfg.body_ids) - 1}")
  tilt = _per_foot_tilt(env, asset_cfg)
  still = _standstill_mask(env, command_name, command_zero_threshold)
  return tilt[:, foot_index] * still.to(tilt.dtype)


def static_double_foot_tilt_multiplier(
  env: "ManagerBasedRlEnv",
  command_name: str,
  command_zero_threshold: float = 0.2,
  asset_cfg: SceneEntityCfg = _DEFAULT_FEET_ASSET_CFG,
) -> torch.Tensor:
  """Return ``exp(-t_left) * exp(-t_right)`` only for static commands."""
  tilt = _per_foot_tilt(env, asset_cfg)
  static_factor = torch.exp(-torch.sum(tilt, dim=1))
  still = _standstill_mask(env, command_name, command_zero_threshold)
  return torch.where(still, static_factor, torch.ones_like(static_factor))


def _foot_contact_force_norms(
  env: "ManagerBasedRlEnv", sensor_name: str
) -> torch.Tensor:
  """Return one terrain contact-force magnitude per configured foot."""
  contact_sensor = env.scene.sensors[sensor_name]
  force = contact_sensor.data.force
  if force is None:
    msg = f"Contact sensor '{sensor_name}' must include force fields."
    raise RuntimeError(msg)
  force_norm = torch.linalg.norm(force, dim=-1)
  while force_norm.ndim > 2:
    force_norm = force_norm.amax(dim=-1)
  return force_norm


def _foot_contact_flags(
  env: "ManagerBasedRlEnv", sensor_name: str, contact_threshold: float
) -> torch.Tensor:
  return _foot_contact_force_norms(env, sensor_name) > contact_threshold


def moving_debounced_support_foot_tilt_penalty(
  env: "ManagerBasedRlEnv",
  command_name: str,
  sensor_name: str,
  contact_threshold: float = 1.0,
  min_contact_time: float = 0.06,
  command_threshold: float = 0.2,
  asset_cfg: SceneEntityCfg = _DEFAULT_FEET_ASSET_CFG,
) -> torch.Tensor:
  """Tilt for moving feet with low-force contact and contact-time debounce."""
  contact_sensor = env.scene.sensors[sensor_name]
  contact_time = contact_sensor.data.current_contact_time
  if contact_time is None:
    msg = f"Contact sensor '{sensor_name}' must enable track_air_time."
    raise RuntimeError(msg)
  in_contact = _foot_contact_flags(env, sensor_name, contact_threshold)
  support = in_contact & (contact_time > min_contact_time)
  tilt = _per_foot_tilt(env, asset_cfg)
  moving = _command_norm(env, command_name) > command_threshold
  return torch.sum(tilt * support.to(tilt.dtype), dim=1) * moving.to(tilt.dtype)


def moving_max_force_foot_tilt_penalty(
  env: "ManagerBasedRlEnv",
  command_name: str,
  sensor_name: str,
  contact_threshold: float = 1.0,
  command_threshold: float = 0.2,
  asset_cfg: SceneEntityCfg = _DEFAULT_FEET_ASSET_CFG,
) -> torch.Tensor:
  """Tilt of the higher-force foot only while a movement command is active."""
  force_norm = _foot_contact_force_norms(env, sensor_name)
  max_force, support_index = torch.max(force_norm, dim=1)
  tilt = _per_foot_tilt(env, asset_cfg)
  support_tilt = torch.gather(tilt, dim=1, index=support_index.unsqueeze(1)).squeeze(1)
  moving = _command_norm(env, command_name) > command_threshold
  has_support = max_force > contact_threshold
  return support_tilt * moving.to(tilt.dtype) * has_support.to(tilt.dtype)


def _rolling_foot_contact_counts(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  contact_threshold: float,
  window_seconds: float,
) -> torch.Tensor:
  """Maintain per-environment left/right contact counts over a fixed window."""
  if window_seconds <= 0.0:
    raise ValueError("window_seconds must be positive")
  window_steps = max(1, round(window_seconds / env.step_dt))
  contact = _foot_contact_flags(env, sensor_name, contact_threshold).to(torch.float32)
  if contact.shape[1] != 2:
    raise ValueError("Exp14 duty balance requires exactly left and right foot contacts")

  state_name = "_exp14_contact_duty_state"
  state = getattr(env, state_name, None)
  if (
    state is None
    or state["history"].shape != (env.num_envs, window_steps, 2)
    or state["history"].device != contact.device
  ):
    state = {
      "history": torch.zeros((env.num_envs, window_steps, 2), device=contact.device),
      "counts": torch.zeros((env.num_envs, 2), device=contact.device),
      "last_episode_length": torch.full(
        (env.num_envs,), -1, device=contact.device, dtype=torch.long
      ),
      "cursor": 0,
    }
    setattr(env, state_name, state)

  episode_length = env.episode_length_buf
  reset = episode_length <= state["last_episode_length"]
  if torch.any(reset):
    state["history"][reset] = 0.0
    state["counts"][reset] = 0.0

  cursor = state["cursor"]
  previous = state["history"][:, cursor].clone()
  state["history"][:, cursor] = contact
  state["counts"] += contact - previous
  state["last_episode_length"] = episode_length.clone()
  state["cursor"] = (cursor + 1) % window_steps
  return state["counts"]


def moving_foot_contact_duty_error(
  env: "ManagerBasedRlEnv",
  command_name: str,
  sensor_name: str,
  contact_threshold: float = 1.0,
  window_seconds: float = 3.0,
  command_threshold: float = 0.2,
) -> torch.Tensor:
  """Left/right 3-second support-duty imbalance while a movement command is active."""
  counts = _rolling_foot_contact_counts(
    env, sensor_name, contact_threshold, window_seconds
  )
  left_share = counts[:, 0] / (torch.sum(counts, dim=1) + 1e-6)
  imbalance = torch.abs(left_share - 0.5)
  moving = _command_norm(env, command_name) > command_threshold
  return imbalance * moving.to(imbalance.dtype)


def moving_foot_contact_duty_multiplier(
  env: "ManagerBasedRlEnv",
  command_name: str,
  sensor_name: str,
  contact_threshold: float = 1.0,
  window_seconds: float = 3.0,
  command_threshold: float = 0.2,
  exp_scale: float = 2.0,
) -> torch.Tensor:
  """Return ``exp(-exp_scale * duty_error)`` for movement, one for static."""
  error = moving_foot_contact_duty_error(
    env,
    command_name=command_name,
    sensor_name=sensor_name,
    contact_threshold=contact_threshold,
    window_seconds=window_seconds,
    command_threshold=command_threshold,
  )
  moving = _command_norm(env, command_name) > command_threshold
  moving_factor = torch.exp(-exp_scale * error)
  return torch.where(moving, moving_factor, torch.ones_like(moving_factor))


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
