"""Curriculum terms for steering-style locomotion tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def body_velocity_command_ranges(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[dict[str, object]],
) -> dict[str, torch.Tensor]:
  """Update a ``BodyVelocityCommandCfg`` range from staged training steps."""
  del env_ids
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = command_term.cfg

  for stage in velocity_stages:
    if env.common_step_counter < int(stage["step"]):
      continue
    if stage.get("lin_vel_x") is not None:
      cfg.lin_vel_x_min, cfg.lin_vel_x_max = stage["lin_vel_x"]
    if stage.get("lin_vel_y") is not None:
      cfg.lin_vel_y_min, cfg.lin_vel_y_max = stage["lin_vel_y"]
    if stage.get("yaw_rate") is not None:
      cfg.yaw_rate_min, cfg.yaw_rate_max = stage["yaw_rate"]

  return {
    "lin_vel_x_min": torch.tensor(cfg.lin_vel_x_min),
    "lin_vel_x_max": torch.tensor(cfg.lin_vel_x_max),
    "lin_vel_y_min": torch.tensor(cfg.lin_vel_y_min),
    "lin_vel_y_max": torch.tensor(cfg.lin_vel_y_max),
    "yaw_rate_min": torch.tensor(cfg.yaw_rate_min),
    "yaw_rate_max": torch.tensor(cfg.yaw_rate_max),
  }


def upstairs_terrain_levels(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  command_name: str,
  success_fraction: float = 0.45,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> dict[str, torch.Tensor]:
  """Progress stair difficulty based on forward travel from the spawn platform."""
  asset = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None
  rel_xy = asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
  distance = torch.norm(rel_xy, dim=-1)

  move_up = distance > terrain_generator.size[0] * success_fraction
  expected_speed = torch.norm(command[env_ids, :2], dim=-1)
  expected_dist = expected_speed * env.max_episode_length_s * 0.4
  move_down = (distance < expected_dist) & ~move_up

  terrain.update_env_origins(env_ids, move_up, move_down)

  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }

  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  if terrain_origins.shape[1] == len(sub_terrain_names):
    for i, name in enumerate(sub_terrain_names):
      mask = terrain.terrain_types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])

  return result
