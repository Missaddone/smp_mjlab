"""Body-velocity command — body-frame xy velocity + yaw rate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


def _xy_world_to_local(vec_w: torch.Tensor, heading_w: torch.Tensor) -> torch.Tensor:
  """Rotate a (N, 2) world-frame xy vector into the heading-aligned frame."""
  cos_h = torch.cos(heading_w)
  sin_h = torch.sin(heading_w)
  x_w, y_w = vec_w[..., 0], vec_w[..., 1]
  return torch.stack([cos_h * x_w + sin_h * y_w, -sin_h * x_w + cos_h * y_w], dim=-1)


def _apply_zero_sampling_and_dead_zone(
  *,
  lin_vel_b: torch.Tensor,
  yaw_rate: torch.Tensor,
  random_values: torch.Tensor,
  zero_command_prob: float,
  dead_zone_speed: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Set body-velocity commands to all-zero by explicit probability or dead zone."""
  zero_mask = torch.zeros_like(yaw_rate, dtype=torch.bool)
  if zero_command_prob > 0.0:
    zero_mask |= random_values < zero_command_prob
  if dead_zone_speed > 0.0:
    command = torch.cat([lin_vel_b, yaw_rate.unsqueeze(-1)], dim=-1)
    zero_mask |= torch.linalg.norm(command, dim=-1) < dead_zone_speed

  lin_vel_b = torch.where(zero_mask.unsqueeze(-1), torch.zeros_like(lin_vel_b), lin_vel_b)
  yaw_rate = torch.where(zero_mask, torch.zeros_like(yaw_rate), yaw_rate)
  return lin_vel_b, yaw_rate


class BodyVelocityCommand(CommandTerm):
  """Periodic body-frame xy velocity and yaw-rate command."""

  cfg: BodyVelocityCommandCfg

  def __init__(self, cfg: BodyVelocityCommandCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.lin_vel_b = torch.zeros(self.num_envs, 2, device=self.device)
    self.yaw_rate = torch.zeros(self.num_envs, device=self.device)
    self.command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_yaw_rate"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.command_b

  def _root_lin_vel_b(self) -> torch.Tensor:
    data = self.robot.data
    if hasattr(data, "root_link_lin_vel_b"):
      return data.root_link_lin_vel_b
    if hasattr(data, "root_lin_vel_b"):
      return data.root_lin_vel_b
    lin_vel_xy_b = _xy_world_to_local(data.root_link_lin_vel_w[:, :2], data.heading_w)
    return torch.cat([lin_vel_xy_b, data.root_link_lin_vel_w[:, 2:3]], dim=-1)

  def _root_yaw_rate(self) -> torch.Tensor:
    data = self.robot.data
    if hasattr(data, "root_link_ang_vel_b"):
      return data.root_link_ang_vel_b[:, 2]
    if hasattr(data, "root_ang_vel_b"):
      return data.root_ang_vel_b[:, 2]
    return data.root_link_ang_vel_w[:, 2]

  def _update_metrics(self) -> None:
    root_lin_vel_b = self._root_lin_vel_b()
    self.metrics["error_vel_xy"] = torch.linalg.norm(
      self.lin_vel_b - root_lin_vel_b[:, :2], dim=-1
    )
    self.metrics["error_yaw_rate"] = torch.abs(self.yaw_rate - self._root_yaw_rate())

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = int(env_ids.numel())
    if n == 0:
      return

    lin_vel_b = torch.empty(n, 2, device=self.device)
    lin_vel_b[:, 0] = torch.empty(n, device=self.device).uniform_(
      self.cfg.lin_vel_x_min, self.cfg.lin_vel_x_max
    )
    lin_vel_b[:, 1] = torch.empty(n, device=self.device).uniform_(
      self.cfg.lin_vel_y_min, self.cfg.lin_vel_y_max
    )

    yaw_rate = torch.empty(n, device=self.device)
    yaw_rate.uniform_(self.cfg.yaw_rate_min, self.cfg.yaw_rate_max)
    lin_vel_b, yaw_rate = _apply_zero_sampling_and_dead_zone(
      lin_vel_b=lin_vel_b,
      yaw_rate=yaw_rate,
      random_values=torch.rand(n, device=self.device),
      zero_command_prob=self.cfg.zero_command_prob,
      dead_zone_speed=self.cfg.dead_zone_speed,
    )
    self.yaw_rate[env_ids] = yaw_rate
    self.lin_vel_b[env_ids] = lin_vel_b
    self.command_b[env_ids, 0:2] = self.lin_vel_b[env_ids]
    self.command_b[env_ids, 2] = self.yaw_rate[env_ids]

  def _update_command(self) -> None:
    self.command_b[:, 0:2] = self.lin_vel_b
    self.command_b[:, 2] = self.yaw_rate

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    heading_ws = self.robot.data.heading_w.cpu().numpy()
    lin_vel_bs = self.lin_vel_b.cpu().numpy()
    yaw_rates = self.yaw_rate.cpu().numpy()
    actual_lin_vel_bs = self._root_lin_vel_b().cpu().numpy()
    actual_yaw_rates = self._root_yaw_rate().cpu().numpy()

    z = float(self.cfg.viz.z_offset)
    actual_z = z + float(self.cfg.viz.actual_z_offset)
    scale = float(self.cfg.viz.scale)
    yaw_scale = float(self.cfg.viz.yaw_scale)
    yaw_radius = float(self.cfg.viz.yaw_radius)

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      heading_w = heading_ws[batch]
      cos_h = math.cos(heading_w)
      sin_h = math.sin(heading_w)
      x_axis_w = np.array([cos_h, sin_h, 0.0])
      y_axis_w = np.array([-sin_h, cos_h, 0.0])
      origin = base_pos_w + np.array([0.0, 0.0, z])
      actual_origin = base_pos_w + np.array([0.0, 0.0, actual_z])

      x_vec = x_axis_w * lin_vel_bs[batch, 0] * scale
      visualizer.add_arrow(
        origin, origin + x_vec, color=(0.0, 0.75, 0.2, 0.75), width=0.015
      )

      actual_x_vec = x_axis_w * actual_lin_vel_bs[batch, 0] * scale
      visualizer.add_arrow(
        actual_origin,
        actual_origin + actual_x_vec,
        color=(0.55, 1.0, 0.55, 0.75),
        width=0.01,
      )

      y_vec = y_axis_w * lin_vel_bs[batch, 1] * scale
      visualizer.add_arrow(
        origin, origin + y_vec, color=(1.0, 0.55, 0.0, 0.75), width=0.015
      )

      actual_y_vec = y_axis_w * actual_lin_vel_bs[batch, 1] * scale
      visualizer.add_arrow(
        actual_origin,
        actual_origin + actual_y_vec,
        color=(1.0, 0.85, 0.35, 0.75),
        width=0.01,
      )

      yaw_start = origin + x_axis_w * yaw_radius + np.array([0.0, 0.0, 0.12])
      yaw_vec = y_axis_w * yaw_rates[batch] * yaw_scale
      visualizer.add_arrow(
        yaw_start, yaw_start + yaw_vec, color=(0.55, 0.15, 1.0, 0.8), width=0.015
      )

      actual_yaw_start = actual_origin + x_axis_w * (yaw_radius + 0.12) + np.array(
        [0.0, 0.0, 0.12]
      )
      actual_yaw_vec = y_axis_w * actual_yaw_rates[batch] * yaw_scale
      visualizer.add_arrow(
        actual_yaw_start,
        actual_yaw_start + actual_yaw_vec,
        color=(0.82, 0.58, 1.0, 0.75),
        width=0.01,
      )


@dataclass(kw_only=True)
class BodyVelocityCommandCfg(CommandTermCfg):
  entity_name: str
  lin_vel_x_min: float = 0.5
  lin_vel_x_max: float = 3.0
  lin_vel_y_min: float = -1.0
  lin_vel_y_max: float = 1.0
  yaw_rate_min: float = -1.0
  yaw_rate_max: float = 1.0
  zero_command_prob: float = 0.0
  dead_zone_speed: float = 0.0

  @dataclass
  class VizCfg:
    z_offset: float = 0.35
    actual_z_offset: float = 0.12
    scale: float = 0.45
    yaw_scale: float = 0.35
    yaw_radius: float = 0.35

  viz: VizCfg = None  # type: ignore[assignment]

  def __post_init__(self) -> None:
    if self.viz is None:
      self.viz = BodyVelocityCommandCfg.VizCfg()
    if self.lin_vel_x_max < self.lin_vel_x_min:
      msg = (
        f"lin_vel_x_max ({self.lin_vel_x_max}) must be >= "
        f"lin_vel_x_min ({self.lin_vel_x_min})."
      )
      raise ValueError(msg)
    if self.lin_vel_y_max < self.lin_vel_y_min:
      msg = (
        f"lin_vel_y_max ({self.lin_vel_y_max}) must be >= "
        f"lin_vel_y_min ({self.lin_vel_y_min})."
      )
      raise ValueError(msg)
    if self.yaw_rate_max < self.yaw_rate_min:
      msg = (
        f"yaw_rate_max ({self.yaw_rate_max}) must be >= "
        f"yaw_rate_min ({self.yaw_rate_min})."
      )
      raise ValueError(msg)
    if not 0.0 <= self.zero_command_prob <= 1.0:
      raise ValueError(
        f"zero_command_prob must be in [0, 1], got {self.zero_command_prob}"
      )
    if self.dead_zone_speed < 0.0:
      raise ValueError(f"dead_zone_speed must be non-negative, got {self.dead_zone_speed}")

  def build(self, env: "ManagerBasedRlEnv") -> BodyVelocityCommand:
    return BodyVelocityCommand(self, env)
