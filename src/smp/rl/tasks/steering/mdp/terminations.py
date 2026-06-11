"""Termination terms for steering-style locomotion tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def root_height_below_origin_offset(
  env: "ManagerBasedRlEnv",
  minimum_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when root height falls below terrain origin plus an offset."""
  asset = env.scene[asset_cfg.name]
  origins = env.scene.env_origins
  return asset.data.root_link_pos_w[:, 2] - origins[:, 2] < minimum_height


def out_of_terrain_bounds(
  env: "ManagerBasedRlEnv",
  margin: float = 0.3,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when the robot leaves its generated terrain patch."""
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_type != "generator":
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  terrain_generator = terrain.cfg.terrain_generator
  if terrain_generator is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  asset = env.scene[asset_cfg.name]
  rel_xy = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  half_x = max(0.0, terrain_generator.size[0] * 0.5 - margin)
  half_y = max(0.0, terrain_generator.size[1] * 0.5 - margin)
  return (rel_xy[:, 0].abs() > half_x) | (rel_xy[:, 1].abs() > half_y)


def terrain_edge_reached(
  env: "ManagerBasedRlEnv",
  threshold_fraction: float = 0.45,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Successful traversal when the robot reaches the commanded stair edge."""
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_type != "generator":
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  terrain_generator = terrain.cfg.terrain_generator
  if terrain_generator is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  asset = env.scene[asset_cfg.name]
  rel_xy = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  distance = torch.norm(rel_xy, dim=-1)
  reached = distance > terrain_generator.size[0] * threshold_fraction
  return reached & (env.episode_length_buf > 2)
