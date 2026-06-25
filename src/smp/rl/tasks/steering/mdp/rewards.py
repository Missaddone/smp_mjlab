"""Steering reward components: linear-velocity tracking + face alignment.

SMP-gated via the generic ``smp.rl.rewards.smp_product``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

from smp.rl.rewards import smp_guidance_reward

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from smp.rl.tasks.steering.mdp.commands import SteeringCommand


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

STANDING_JOINT_TARGET: tuple[float, ...] = (
  -0.10,
  0.0,
  0.0,
  0.30,
  -0.20,
  0.0,
  -0.10,
  0.0,
  0.0,
  0.30,
  -0.20,
  0.0,
  0.0,
  0.0,
  0.0,
  0.20,
  0.15,
  0.0,
  0.60,
  0.0,
  0.0,
  0.0,
  0.20,
  -0.15,
  0.0,
  0.60,
  0.0,
  0.0,
  0.0,
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


def _command_deadzone_bounds(command) -> tuple[float, float]:
  deadzone_min = getattr(command.cfg, "speed_deadzone_min", None)
  deadzone_max = getattr(command.cfg, "speed_deadzone_max", None)
  if deadzone_min is None and deadzone_max is None:
    deadzone_min = getattr(command.cfg, "tar_speed_min", float("-inf"))
    deadzone_max = getattr(command.cfg, "speed_deadzone", float("-inf"))
  elif deadzone_min is None:
    deadzone_min = getattr(command.cfg, "tar_speed_min", float("-inf"))
  elif deadzone_max is None:
    deadzone_max = getattr(command.cfg, "speed_deadzone", float("-inf"))
  return deadzone_min, deadzone_max


def _command_standing_mask(command) -> torch.Tensor:
  run_speed_min = getattr(command.cfg, "run_speed_min", None)
  run_speed_max = getattr(command.cfg, "run_speed_max", None)
  if (
    getattr(command.cfg, "stand_sample_prob", 0.0) > 0.0
    and run_speed_min is not None
    and run_speed_max is not None
  ):
    stand_speed = getattr(command.cfg, "stand_speed", 0.0)
    tolerance = getattr(command.cfg, "stand_speed_tolerance", 1e-4)
    return torch.abs(command.tar_speed - stand_speed) <= tolerance

  deadzone_min, deadzone_max = _command_deadzone_bounds(command)
  return (command.tar_speed >= deadzone_min) & (command.tar_speed <= deadzone_max)


def _command_moving_mask(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  return ~_command_standing_mask(command)


def _standing_joint_target_tensor(env: "ManagerBasedRlEnv", dtype: torch.dtype) -> torch.Tensor:
  target = getattr(env, "_smp_standing_joint_target", None)
  if target is None or target.device != torch.device(env.device) or target.dtype != dtype:
    target = torch.tensor(STANDING_JOINT_TARGET, dtype=dtype, device=env.device)
    env._smp_standing_joint_target = target  # type: ignore[attr-defined]
  return target


def base_upright_penalty(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def root_height_below_target_penalty(
  env: "ManagerBasedRlEnv",
  target_height: float = 0.74,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  root_height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return torch.square((target_height - root_height).clamp(min=0.0))


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


def upstairs_height_progress(
  env: "ManagerBasedRlEnv",
  target_climb_height: float = 0.8,
  nominal_root_height: float = 0.75,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward climbing above the terrain spawn platform.

  The terrain origin is the stair-bottom platform, while the robot root starts
  roughly ``nominal_root_height`` above that platform.  Subtracting the nominal
  standing height makes the reward measure stair-climb progress rather than the
  robot's ordinary base height.
  """
  asset = env.scene[asset_cfg.name]
  origins = env.scene.env_origins
  climb = asset.data.root_link_pos_w[:, 2] - origins[:, 2] - nominal_root_height
  return (climb / target_climb_height).clamp(0.0, 1.0)


def standing_pose_reward(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  target_height: float = 0.79,
  height_std: float = 0.08,
  joint_std: float = 0.12,
  upright_scale: float = 8.0,
  lin_vel_scale: float = 4.0,
  ang_vel_scale: float = 1.0,
  joint_vel_scale: float = 0.05,
  action_scale: float = 0.1,
) -> torch.Tensor:
  """Explicit quiet standing reward for low-speed command branches."""
  asset = env.scene[asset_cfg.name]
  data = asset.data
  root_height = data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  joint_count = len(STANDING_JOINT_TARGET)
  joint_pos = data.joint_pos[:, :joint_count]
  joint_vel = data.joint_vel[:, :joint_count]
  standing_target = _standing_joint_target_tensor(env, data.joint_pos.dtype)
  joint_err = joint_pos - standing_target
  action = env.action_manager.action

  height = torch.exp(-((root_height - target_height) / height_std) ** 2)
  pose = torch.exp(-torch.mean((joint_err / joint_std) ** 2, dim=-1))
  upright = torch.exp(-upright_scale * torch.sum(data.projected_gravity_b[:, :2] ** 2, dim=-1))
  lin_quiet = torch.exp(-lin_vel_scale * torch.sum(_root_lin_vel_w(data)[:, :2] ** 2, dim=-1))
  ang_quiet = torch.exp(-ang_vel_scale * torch.sum(_root_ang_vel_w(data) ** 2, dim=-1))
  joint_quiet = torch.exp(-joint_vel_scale * torch.mean(joint_vel**2, dim=-1))
  action_regularization = torch.exp(-action_scale * torch.mean(action**2, dim=-1))

  return (
    0.20 * height
    + 1.00 * pose
    + 0.20 * upright
    + 0.08 * lin_quiet
    + 0.05 * ang_quiet
    + 0.04 * joint_quiet
    + 0.03 * action_regularization
  )


def forward_task_smp_product(
  env: "ManagerBasedRlEnv",
  command_name: str = "steering",
  vel_err_scale: float = 0.5,
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 6.0,
) -> torch.Tensor:
  task = steering_target_velocity(env, command_name=command_name, vel_err_scale=vel_err_scale)
  return task * smp_guidance_reward(env, fixed_timesteps=fixed_timesteps, ws=ws)


def steering_task_smp_product(
  env: "ManagerBasedRlEnv",
  command_name: str = "steering",
  vel_err_scale: float = 1.0,
  velocity_weight: float = 0.5,
  face_weight: float = 0.5,
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 6.0,
) -> torch.Tensor:
  velocity = steering_target_velocity(env, command_name=command_name, vel_err_scale=vel_err_scale)
  face = steering_face_direction(env, command_name=command_name)
  task = velocity_weight * velocity + face_weight * face
  return task * smp_guidance_reward(env, fixed_timesteps=fixed_timesteps, ws=ws)


def steering_modified_stand_branch_reward(
  env: "ManagerBasedRlEnv",
  command_name: str = "steering",
  vel_err_scale: float = 1.0,
  velocity_weight: float = 1.0,
  face_weight: float = 0.5,
  deadzone_face_weight: float = 0.0,
  deadzone_stand_weight: float = 0.5,
  deadzone_lin_vel_penalty_weight: float = 2.0,
  deadzone_joint_vel_penalty_weight: float = 0.05,
  deadzone_action_penalty_weight: float = 0.05,
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 6.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  moving_mask = _command_moving_mask(env, command_name)
  velocity = steering_target_velocity(env, command_name=command_name, vel_err_scale=vel_err_scale)
  face = steering_face_direction(env, command_name=command_name)
  task = velocity_weight * velocity + face_weight * face

  asset = env.scene[asset_cfg.name]
  joint_count = len(STANDING_JOINT_TARGET)
  root_speed_sq = torch.sum(_root_lin_vel_w(asset.data)[:, :2] ** 2, dim=-1)
  joint_vel_sq = torch.mean(asset.data.joint_vel[:, :joint_count] ** 2, dim=-1)
  action_sq = torch.mean(env.action_manager.action**2, dim=-1)
  stand = (
    velocity_weight * velocity
    + deadzone_face_weight * face
    + deadzone_stand_weight * standing_pose_reward(env, asset_cfg=asset_cfg)
  )
  stand = stand - deadzone_lin_vel_penalty_weight * root_speed_sq
  stand = stand - deadzone_joint_vel_penalty_weight * joint_vel_sq
  stand = stand - deadzone_action_penalty_weight * action_sq
  stand = torch.clamp(stand, min=0.0)

  moving = task * smp_guidance_reward(env, fixed_timesteps=fixed_timesteps, ws=ws, env_mask=moving_mask)
  return torch.where(moving_mask, moving, stand)


def body_velocity_task_smp_product(
  env: "ManagerBasedRlEnv",
  command_name: str = "steering",
  lin_vel_err_scale: float = 2.0,
  yaw_rate_err_scale: float = 1.0,
  lin_vel_weight: float = 0.75,
  yaw_rate_weight: float = 0.25,
  zero_lin_vel_target: bool = False,
  stand_speed_threshold: float = 0.2,
  stand_yaw_rate_threshold: float = 0.2,
  use_stand_branch: bool = True,
  style_floor: float = 0.0,
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 6.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  if not 0.0 <= style_floor <= 1.0:
    msg = f"style_floor must be in [0, 1], got {style_floor}."
    raise ValueError(msg)
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)
  root_lin_vel_b = _root_lin_vel_b(asset.data)
  target_lin_vel_b = torch.zeros_like(command.lin_vel_b) if zero_lin_vel_target else command.lin_vel_b
  lin_vel_err = torch.sum((root_lin_vel_b[:, :2] - target_lin_vel_b) ** 2, dim=-1)
  lin_vel_reward = torch.exp(-lin_vel_err_scale * lin_vel_err)
  yaw_rate_err = (command.yaw_rate - _root_yaw_rate(asset.data)) ** 2
  yaw_rate_reward = torch.exp(-yaw_rate_err_scale * yaw_rate_err)

  task = lin_vel_weight * lin_vel_reward + yaw_rate_weight * yaw_rate_reward
  style = smp_guidance_reward(env, fixed_timesteps=fixed_timesteps, ws=ws)
  moving_reward = task * (style_floor + (1.0 - style_floor) * style)
  if not use_stand_branch:
    return moving_reward

  cmd_speed = torch.linalg.norm(command.lin_vel_b, dim=-1)
  stand_mask = (cmd_speed < stand_speed_threshold) & (torch.abs(command.yaw_rate) < stand_yaw_rate_threshold)
  stand_reward = standing_pose_reward(env, asset_cfg=asset_cfg)
  return torch.where(stand_mask, stand_reward, moving_reward)


def steering_doubleprior_task_reward(
  env: "ManagerBasedRlEnv",
  command_name: str = "steering",
  vel_err_scale: float = 1.0,
  velocity_weight: float = 1.0,
  face_weight: float = 0.5,
  deadzone_face_weight: float = 0.0,
  deadzone_stand_weight: float = 0.5,
  deadzone_lin_vel_penalty_weight: float = 2.0,
  deadzone_joint_vel_penalty_weight: float = 0.05,
  deadzone_action_penalty_weight: float = 0.05,
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  moving_ws: float = 6.0,
  stand_ws: float = 6.0,
  moving_prior_name: str = "moving",
  stand_prior_name: str = "stand",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  moving_mask = _command_moving_mask(env, command_name)
  velocity = steering_target_velocity(env, command_name=command_name, vel_err_scale=vel_err_scale)
  face = steering_face_direction(env, command_name=command_name)
  moving_task = velocity_weight * velocity + face_weight * face

  asset = env.scene[asset_cfg.name]
  joint_count = len(STANDING_JOINT_TARGET)
  root_speed_sq = torch.sum(_root_lin_vel_w(asset.data)[:, :2] ** 2, dim=-1)
  joint_vel_sq = torch.mean(asset.data.joint_vel[:, :joint_count] ** 2, dim=-1)
  action_sq = torch.mean(env.action_manager.action**2, dim=-1)
  stand_task = (
    velocity_weight * velocity
    + deadzone_face_weight * face
    + deadzone_stand_weight * standing_pose_reward(env, asset_cfg=asset_cfg)
  )
  stand_task = stand_task - deadzone_lin_vel_penalty_weight * root_speed_sq
  stand_task = stand_task - deadzone_joint_vel_penalty_weight * joint_vel_sq
  stand_task = stand_task - deadzone_action_penalty_weight * action_sq
  stand_task = torch.clamp(stand_task, min=0.0)

  moving_style = smp_guidance_reward(
    env,
    fixed_timesteps=fixed_timesteps,
    ws=moving_ws,
    env_mask=moving_mask,
    prior_name=moving_prior_name,
  )
  stand_style = smp_guidance_reward(
    env,
    fixed_timesteps=fixed_timesteps,
    ws=stand_ws,
    env_mask=~moving_mask,
    prior_name=stand_prior_name,
  )
  return torch.where(moving_mask, moving_task * moving_style, stand_task * stand_style)


def filtered_contact_force_penalty(
  env: "ManagerBasedRlEnv",
  sensor_cfg: SceneEntityCfg,
  threshold: float = 2.0,
  saturation_force: float = 20.0,
) -> torch.Tensor:
  contact_sensor = env.scene.sensors[sensor_cfg.name]
  force_history = contact_sensor.data.force_history
  force = contact_sensor.data.force if force_history is None else force_history
  if force is None:
    msg = f"Contact sensor '{sensor_cfg.name}' must include force fields."
    raise RuntimeError(msg)
  if saturation_force <= threshold:
    msg = "saturation_force must be greater than threshold."
    raise ValueError(msg)
  contact_force = torch.linalg.norm(force, dim=-1).amax(dim=tuple(range(1, force.ndim - 1)))
  return ((contact_force - threshold) / (saturation_force - threshold)).clamp(0.0, 1.0)


def feet_slide_penalty(
  env: "ManagerBasedRlEnv",
  sensor_cfg: SceneEntityCfg,
  asset_cfg: SceneEntityCfg,
  contact_threshold: float = 1.0,
) -> torch.Tensor:
  contact_sensor = env.scene.sensors[sensor_cfg.name]
  force = contact_sensor.data.force
  if force is None:
    msg = f"Contact sensor '{sensor_cfg.name}' must include force fields."
    raise RuntimeError(msg)
  in_contact = torch.linalg.norm(force, dim=-1) > contact_threshold
  asset = env.scene[asset_cfg.name]
  body_vel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids]
  return torch.sum(torch.linalg.norm(body_vel_w[..., :2], dim=-1) * in_contact, dim=1)


def support_foot_tilt_penalty(
  env: "ManagerBasedRlEnv",
  sensor_cfg: SceneEntityCfg,
  asset_cfg: SceneEntityCfg,
  contact_threshold: float = 1.0,
) -> torch.Tensor:
  contact_sensor = env.scene.sensors[sensor_cfg.name]
  force = contact_sensor.data.force
  if force is None:
    msg = f"Contact sensor '{sensor_cfg.name}' must include force fields."
    raise RuntimeError(msg)
  in_contact = torch.linalg.norm(force, dim=-1) > contact_threshold
  asset = env.scene[asset_cfg.name]
  body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
  local_up = torch.zeros((*body_quat_w.shape[:-1], 3), device=body_quat_w.device, dtype=body_quat_w.dtype)
  local_up[..., 2] = 1.0
  foot_up_w = quat_apply(body_quat_w.reshape(-1, 4), local_up.reshape(-1, 3)).reshape_as(local_up)
  tilt = torch.sum(torch.square(foot_up_w[..., :2]), dim=-1)
  return torch.sum(tilt * in_contact, dim=1)


def biped_feet_air_time_reward(
  env: "ManagerBasedRlEnv",
  command_name: str,
  sensor_cfg: SceneEntityCfg,
  threshold: float = 0.4,
  command_speed_threshold: float = 0.15,
) -> torch.Tensor:
  contact_sensor = env.scene.sensors[sensor_cfg.name]
  air_time = contact_sensor.data.current_air_time
  contact_time = contact_sensor.data.current_contact_time
  if air_time is None or contact_time is None:
    msg = f"Contact sensor '{sensor_cfg.name}' must enable track_air_time."
    raise RuntimeError(msg)
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.sum(in_contact, dim=1) == 1
  reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1).values
  reward = torch.clamp(reward, max=threshold)
  command_speed = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
  return reward * (command_speed > command_speed_threshold)


def persistent_single_support_penalty(
  env: "ManagerBasedRlEnv",
  command_name: str,
  sensor_cfg: SceneEntityCfg,
  max_single_support_time: float = 0.6,
  max_excess_time: float = 1.0,
  command_speed_threshold: float = 0.15,
) -> torch.Tensor:
  contact_sensor = env.scene.sensors[sensor_cfg.name]
  contact_time = contact_sensor.data.current_contact_time
  if contact_time is None:
    msg = f"Contact sensor '{sensor_cfg.name}' must enable track_air_time."
    raise RuntimeError(msg)
  in_contact = contact_time > 0.0
  single_stance = torch.sum(in_contact, dim=1) == 1
  support_time = torch.max(contact_time, dim=1).values
  excess = (support_time - max_single_support_time).clamp(min=0.0, max=max_excess_time)
  command_speed = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
  return excess * single_stance * (command_speed > command_speed_threshold)
