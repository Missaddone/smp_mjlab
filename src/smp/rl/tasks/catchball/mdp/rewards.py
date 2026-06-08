"""Reward components for the goalkeeper task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from smp.rl.tasks.catchball.mdp.commands import CatchBallCommand


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def goalkeeper_block_shot(
  env: "ManagerBasedRlEnv",
  command_name: str,
  catch_distance: float = 0.18,
  proximity_scale: float = 18.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward blocking the active shot.

  This is intentionally sparse: no proximity shaping, no reward for chasing the
  ball, only real robot-ball contact while the shot is active.
  """
  asset = env.scene[asset_cfg.name]
  cmd: "CatchBallCommand" = env.command_manager.get_term(command_name)  # type: ignore[assignment]

  del asset, proximity_scale
  return cmd.block_contact(catch_distance).float()


def catch_ball_contact(
  env: "ManagerBasedRlEnv",
  command_name: str,
  catch_distance: float = 0.18,
  proximity_scale: float = 18.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Backward-compatible alias for the goalkeeper block reward."""
  return goalkeeper_block_shot(
    env,
    command_name=command_name,
    catch_distance=catch_distance,
    proximity_scale=proximity_scale,
    asset_cfg=asset_cfg,
  )


def catch_ball_face_direction(
  env: "ManagerBasedRlEnv",
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """``max(face_dir · char_face_dir, 0)`` — both unit world-xy vectors."""
  asset = env.scene[asset_cfg.name]
  cmd: "CatchBallCommand" = env.command_manager.get_term(command_name)  # type: ignore[assignment]

  heading_w = asset.data.heading_w
  char_face_w = torch.stack([torch.cos(heading_w), torch.sin(heading_w)], dim=-1)
  face_dot = (cmd.face_dir_w * char_face_w).sum(dim=-1)
  return face_dot.clamp_min(0.0)


def goalkeeper_stay_on_line(
  env: "ManagerBasedRlEnv",
  command_name: str,
  distance_from_goal: float = 0.5,
  line_err_scale: float = 10.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward staying on a plane parallel to the goal line.

  The robot is only penalized for moving too far forward/backward from the
  goalkeeper line; lateral movement along the goal mouth is left free.
  """
  asset = env.scene[asset_cfg.name]
  cmd: "CatchBallCommand" = env.command_manager.get_term(command_name)  # type: ignore[assignment]

  goal_center_xy = cmd.goal_center_w()[:, :2]
  target_xy = goal_center_xy - cmd.goal_dir_w * distance_from_goal
  root_xy = asset.data.root_link_pos_w[:, :2]
  signed_err = ((root_xy - target_xy) * cmd.goal_dir_w).sum(dim=-1)
  return torch.exp(-line_err_scale * signed_err.square())


def goalkeeper_position_penalty(
  env: "ManagerBasedRlEnv",
  command_name: str,
  distance_from_goal: float = 0.5,
  forward_scale: float = 6.0,
  lateral_scale: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalty for leaving the preferred goalkeeper position.

  The target is the center of the goal mouth, shifted into the field by
  ``distance_from_goal``. Forward/backward error is penalized more strongly than
  lateral error, so the robot prefers the center but can still move sideways.
  """
  asset = env.scene[asset_cfg.name]
  cmd: "CatchBallCommand" = env.command_manager.get_term(command_name)  # type: ignore[assignment]

  goal_center_xy = cmd.goal_center_w()[:, :2]
  target_xy = goal_center_xy - cmd.goal_dir_w * distance_from_goal
  root_delta = asset.data.root_link_pos_w[:, :2] - target_xy

  forward_err = (root_delta * cmd.goal_dir_w).sum(dim=-1)
  lateral_err = (root_delta * cmd.goal_left_w).sum(dim=-1)
  return forward_scale * forward_err.square() + lateral_scale * lateral_err.square()


def goalkeeper_goal_allowed_penalty(
  env: "ManagerBasedRlEnv",
  command_name: str,
) -> torch.Tensor:
  """Penalty of ``1`` when the active ball enters the goal mouth."""
  cmd: "CatchBallCommand" = env.command_manager.get_term(command_name)  # type: ignore[assignment]
  return cmd.ball_active.float() * cmd.ball_crossed_goal_line().float()
