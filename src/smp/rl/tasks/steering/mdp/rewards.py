"""Steering reward components: linear-velocity tracking + face alignment.

SMP-gated via the generic ``smp.rl.rewards.smp_product``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply
from mjlab.utils.lab_api.string import resolve_matching_names_values

from smp.rl.rewards import _cache_smp_reward_components, smp_guidance_reward

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from smp.rl.tasks.steering.mdp.commands import SteeringCommand


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


def base_upright_penalty(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def zero_command_action_rate_l2(
  env: "ManagerBasedRlEnv",
  command_name: str = "steering",
  command_threshold: float = 0.1,
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  if command is None:
    return torch.zeros(env.num_envs, device=env.device)
  linear_norm = torch.linalg.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  zero_command = (linear_norm + angular_norm) <= command_threshold
  action_rate = torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  )
  return action_rate * zero_command.float()


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


def steering_signed_target_velocity(
  env: "ManagerBasedRlEnv",
  command_name: str,
  vel_err_scale: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Signed speed velocity tracking without filtering negative projection.

  This is intended for fixed-direction commands that may sample negative speed:
  a negative target speed should reward moving backward along ``tar_dir_w``.
  """
  asset = env.scene[asset_cfg.name]
  cmd: "SteeringCommand" = env.command_manager.get_term(command_name)  # type: ignore[assignment]

  root_vel_xy = asset.data.root_link_lin_vel_w[:, :2]
  tar_vel = cmd.tar_speed.unsqueeze(-1) * cmd.tar_dir_w
  vel_err = ((tar_vel - root_vel_xy) ** 2).sum(dim=-1)
  return torch.exp(-vel_err_scale * vel_err)


def signed_forward_task_smp_blend(
  env: "ManagerBasedRlEnv",
  command_name: str = "steering",
  vel_err_scale: float = 0.5,
  style_floor: float = 0.3,
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 6.0,
) -> torch.Tensor:
  if not 0.0 <= style_floor <= 1.0:
    msg = f"style_floor must be in [0, 1], got {style_floor}."
    raise ValueError(msg)
  task = steering_signed_target_velocity(
    env, command_name=command_name, vel_err_scale=vel_err_scale
  )
  style = smp_guidance_reward(env, fixed_timesteps=fixed_timesteps, ws=ws)
  product = task * (style_floor + (1.0 - style_floor) * style)
  _cache_smp_reward_components(env, task, style, product)
  return product


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


class speed_dependent_joint_posture_reward:
  """Reward staying near default joint pose with speed-dependent tolerance."""

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    asset = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(asset_cfg.joint_names)
    _, _, std_slow = resolve_matching_names_values(
      data=cfg.params["std_slow"],
      list_of_strings=joint_names,
    )
    _, _, std_fast = resolve_matching_names_values(
      data=cfg.params["std_fast"],
      list_of_strings=joint_names,
    )
    self.std_slow = torch.tensor(std_slow, device=env.device, dtype=torch.float32)
    self.std_fast = torch.tensor(std_fast, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str,
    std_slow: dict[str, float],
    std_fast: dict[str, float],
    fast_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_slow, std_fast
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.linalg.norm(command[:, :2], dim=-1)
    yaw_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + yaw_speed
    fast_mask = total_speed >= fast_threshold
    std = torch.where(fast_mask.unsqueeze(-1), self.std_fast, self.std_slow)

    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    err = joint_pos - default_joint_pos
    return torch.exp(-torch.mean(err**2 / (std**2), dim=-1))


def forward_task_smp_product(
  env: "ManagerBasedRlEnv",
  command_name: str = "steering",
  vel_err_scale: float = 0.5,
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 6.0,
) -> torch.Tensor:
  task = steering_target_velocity(env, command_name=command_name, vel_err_scale=vel_err_scale)
  style = smp_guidance_reward(env, fixed_timesteps=fixed_timesteps, ws=ws)
  product = task * style
  _cache_smp_reward_components(env, task, style, product)
  return product


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
  style = smp_guidance_reward(env, fixed_timesteps=fixed_timesteps, ws=ws)
  product = task * style
  _cache_smp_reward_components(env, task, style, product)
  return product


def body_velocity_task_smp_product(
  env: "ManagerBasedRlEnv",
  command_name: str = "steering",
  lin_vel_err_scale: float = 2.0,
  yaw_rate_err_scale: float = 1.0,
  lin_vel_weight: float = 0.75,
  yaw_rate_weight: float = 0.25,
  zero_lin_vel_target: bool = False,
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
  product = task * (style_floor + (1.0 - style_floor) * style)
  _cache_smp_reward_components(env, task, style, product)
  return product


def body_velocity_task_smp_sum(
  env: "ManagerBasedRlEnv",
  command_name: str = "steering",
  lin_vel_err_scale: float = 2.0,
  yaw_rate_err_scale: float = 1.0,
  lin_vel_weight: float = 0.75,
  yaw_rate_weight: float = 0.25,
  task_weight: float = 1.0,
  style_weight: float = 1.0,
  zero_lin_vel_target: bool = False,
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 6.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
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
  total = task_weight * task + style_weight * style
  _cache_smp_reward_components(env, task, style, total)
  return total


def body_velocity_task_component(
  env: "ManagerBasedRlEnv",
  command_name: str = "steering",
  lin_vel_err_scale: float = 2.0,
  yaw_rate_err_scale: float = 1.0,
  lin_vel_weight: float = 0.75,
  yaw_rate_weight: float = 0.25,
  zero_lin_vel_target: bool = False,
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 6.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
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
  product = task * style
  _cache_smp_reward_components(env, task, style, product)
  return task


def smp_cached_component(
  env: "ManagerBasedRlEnv",
  component: str,
) -> torch.Tensor:
  cached = getattr(env, "_smp_reward_components", None)
  if cached is None or component not in cached:
    return torch.zeros(env.num_envs, device=env.device)
  return cached[component]


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
