"""Steering command — target xy direction + speed + face direction.

Each env carries a periodically-resampled world-frame target dir, speed, and
face dir; the exposed command is in the robot's local heading frame, so the
observation is yaw-invariant.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  import viser
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


def _dir_world_to_local(dir_w: torch.Tensor, heading_w: torch.Tensor) -> torch.Tensor:
  """Rotate a (N, 2) world-frame xy direction into the heading-aligned frame."""
  cos_h = torch.cos(heading_w)
  sin_h = torch.sin(heading_w)
  x_w, y_w = dir_w[..., 0], dir_w[..., 1]
  return torch.stack([cos_h * x_w + sin_h * y_w, -sin_h * x_w + cos_h * y_w], dim=-1)


def _deadzone_bounds(cfg: "SteeringCommandCfg") -> tuple[float, float]:
  lower = cfg.speed_deadzone_min
  upper = cfg.speed_deadzone_max
  if lower is None and upper is None:
    lower = cfg.tar_speed_min
    upper = cfg.speed_deadzone
  elif lower is None:
    lower = cfg.tar_speed_min
  elif upper is None:
    upper = cfg.speed_deadzone
  return lower, upper


def _has_two_bin_speed_sampling(cfg: "SteeringCommandCfg") -> bool:
  return (
    cfg.stand_sample_prob > 0.0
    and cfg.run_speed_min is not None
    and cfg.run_speed_max is not None
  )


class SteeringCommand(CommandTerm):
  """Periodic target dir + speed + face dir command (world frame internally)."""

  cfg: SteeringCommandCfg

  def __init__(self, cfg: SteeringCommandCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    # World-frame state.
    self.tar_dir_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.face_dir_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.tar_speed = torch.zeros(self.num_envs, device=self.device)
    self.tar_dir_w[..., 0] = 1.0
    self.face_dir_w[..., 0] = 1.0

    # Heading-frame command exposed to the policy: [tar_dir_x, tar_dir_y,
    # tar_speed, face_dir_x, face_dir_y].
    self.command_b = torch.zeros(self.num_envs, 5, device=self.device)

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_face"] = torch.zeros(self.num_envs, device=self.device)

    # Set by create_gui() when the viewer is active.
    self._gui_enabled: viser.GuiCheckboxHandle | None = None
    self._gui_speed: viser.GuiSliderHandle | None = None
    self._gui_tar_angle: viser.GuiSliderHandle | None = None
    self._gui_face_angle: viser.GuiSliderHandle | None = None
    self._gui_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.command_b

  def _update_metrics(self) -> None:
    max_step = self.cfg.resampling_time_range[1] / self._env.step_dt
    tar_vel_w = self.tar_speed.unsqueeze(-1) * self.tar_dir_w
    vel_err = torch.norm(tar_vel_w - self.robot.data.root_link_lin_vel_w[:, :2], dim=-1)
    self.metrics["error_vel_xy"] += vel_err / max_step

    heading_w = self.robot.data.heading_w
    char_face_w = torch.stack([torch.cos(heading_w), torch.sin(heading_w)], dim=-1)
    face_dot = (self.face_dir_w * char_face_w).sum(dim=-1)
    self.metrics["error_face"] += (1.0 - face_dot.clamp(-1.0, 1.0)) / max_step

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = int(env_ids.numel())
    r = torch.empty(n, device=self.device)

    if self.cfg.rand_tar_dir:
      theta = r.uniform_(-math.pi, math.pi)
    else:
      theta = torch.zeros(n, device=self.device)
    self.tar_dir_w[env_ids, 0] = torch.cos(theta)
    self.tar_dir_w[env_ids, 1] = torch.sin(theta)

    if _has_two_bin_speed_sampling(self.cfg):
      run_speed_min = max(float(self.cfg.run_speed_min), self.cfg.tar_speed_min)
      run_speed_max = min(float(self.cfg.run_speed_max), self.cfg.tar_speed_max)
      if run_speed_max <= run_speed_min:
        msg = f"Invalid run speed interval [{run_speed_min}, {run_speed_max}]."
        raise ValueError(msg)
      stand_mask = torch.rand(n, device=self.device) < self.cfg.stand_sample_prob
      speeds = torch.empty(n, device=self.device)
      if stand_mask.any():
        speeds[stand_mask] = self.cfg.stand_speed
      if (~stand_mask).any():
        speeds[~stand_mask] = torch.empty(int((~stand_mask).sum()), device=self.device).uniform_(
          run_speed_min, run_speed_max
        )
      self.tar_speed[env_ids] = speeds
    else:
      deadzone_min, deadzone_max = _deadzone_bounds(self.cfg)
      deadzone_min = max(deadzone_min, self.cfg.tar_speed_min)
      deadzone_max = min(deadzone_max, self.cfg.tar_speed_max)
      has_deadzone = deadzone_max > deadzone_min
      if self.cfg.deadzone_sample_prob > 0.0 and has_deadzone:
        deadzone_mask = torch.rand(n, device=self.device) < self.cfg.deadzone_sample_prob
        speeds = torch.empty(n, device=self.device)
        if deadzone_mask.any():
          speeds[deadzone_mask] = torch.empty(int(deadzone_mask.sum()), device=self.device).uniform_(
            deadzone_min, deadzone_max
          )
        if (~deadzone_mask).any():
          moving_count = int((~deadzone_mask).sum())
          moving_speeds = torch.empty(moving_count, device=self.device)
          lower_len = max(deadzone_min - self.cfg.tar_speed_min, 0.0)
          upper_len = max(self.cfg.tar_speed_max - deadzone_max, 0.0)
          if lower_len == 0.0 and upper_len == 0.0:
            moving_speeds.uniform_(deadzone_min, deadzone_max)
          elif lower_len > 0.0 and upper_len > 0.0:
            lower_mask = torch.rand(moving_count, device=self.device) < lower_len / (
              lower_len + upper_len
            )
            if lower_mask.any():
              moving_speeds[lower_mask] = torch.empty(
                int(lower_mask.sum()), device=self.device
              ).uniform_(self.cfg.tar_speed_min, deadzone_min)
            if (~lower_mask).any():
              moving_speeds[~lower_mask] = torch.empty(
                int((~lower_mask).sum()), device=self.device
              ).uniform_(deadzone_max, self.cfg.tar_speed_max)
          elif lower_len > 0.0:
            moving_speeds.uniform_(self.cfg.tar_speed_min, deadzone_min)
          else:
            moving_speeds.uniform_(deadzone_max, self.cfg.tar_speed_max)
          speeds[~deadzone_mask] = moving_speeds
        self.tar_speed[env_ids] = speeds
      else:
        self.tar_speed[env_ids] = torch.empty(n, device=self.device).uniform_(
          self.cfg.tar_speed_min, self.cfg.tar_speed_max
        )

    if self.cfg.rand_face_dir:
      face_theta = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
    else:
      face_theta = theta
    self.face_dir_w[env_ids, 0] = torch.cos(face_theta)
    self.face_dir_w[env_ids, 1] = torch.sin(face_theta)

  def _update_command(self) -> None:
    heading_w = self.robot.data.heading_w
    self.command_b[:, 0:2] = _dir_world_to_local(self.tar_dir_w, heading_w)
    self.command_b[:, 2] = self.tar_speed
    self.command_b[:, 3:5] = _dir_world_to_local(self.face_dir_w, heading_w)

  # GUI.

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create steering joystick sliders in the Viser viewer."""
    from viser import Icon

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)
      speed_slider = server.gui.add_slider(
        "tar_speed",
        min=float(self.cfg.tar_speed_min),
        max=float(self.cfg.tar_speed_max),
        step=0.1,
        initial_value=max(float(self.cfg.tar_speed_min), min(1.0, float(self.cfg.tar_speed_max))),
      )
      tar_angle_slider = server.gui.add_slider(
        "tar_angle (rad)",
        min=-math.pi,
        max=math.pi,
        step=0.05,
        initial_value=0.0,
      )
      face_angle_slider = server.gui.add_slider(
        "face_angle (rad)",
        min=-math.pi,
        max=math.pi,
        step=0.05,
        initial_value=0.0,
      )
      zero_btn = server.gui.add_button("Zero speed", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        speed_slider.value = 0.0

    self._gui_enabled = enabled
    self._gui_speed = speed_slider
    self._gui_tar_angle = tar_angle_slider
    self._gui_face_angle = face_angle_slider
    self._gui_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._gui_enabled is None or not self._gui_enabled.value:
      return
    assert self._gui_get_env_idx is not None
    assert self._gui_speed is not None
    assert self._gui_tar_angle is not None
    assert self._gui_face_angle is not None
    idx = self._gui_get_env_idx()
    tar_a = float(self._gui_tar_angle.value)
    face_a = float(self._gui_face_angle.value)
    self.tar_dir_w[idx, 0] = math.cos(tar_a)
    self.tar_dir_w[idx, 1] = math.sin(tar_a)
    self.face_dir_w[idx, 0] = math.cos(face_a)
    self.face_dir_w[idx, 1] = math.sin(face_a)
    self.tar_speed[idx] = float(self._gui_speed.value)
    # Refresh heading-frame command so the override shows up in the obs.
    self._update_command()

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    tar_dir_ws = self.tar_dir_w.cpu().numpy()
    tar_speed_s = self.tar_speed.cpu().numpy()
    face_dir_ws = self.face_dir_w.cpu().numpy()
    lin_vel_ws = self.robot.data.root_link_lin_vel_w.cpu().numpy()

    z = float(self.cfg.viz.z_offset)
    scale = float(self.cfg.viz.scale)

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      origin = base_pos_w + np.array([0.0, 0.0, z])

      # Commanded velocity arrow (blue): tar_dir * tar_speed in world frame.
      cmd_vec = (
        np.array(
          [
            tar_dir_ws[batch, 0] * tar_speed_s[batch],
            tar_dir_ws[batch, 1] * tar_speed_s[batch],
            0.0,
          ]
        )
        * scale
      )
      visualizer.add_arrow(
        origin, origin + cmd_vec, color=(0.2, 0.2, 0.6, 0.6), width=0.015
      )

      # Face direction arrow (red): unit-length world-frame face_dir.
      face_vec = np.array([face_dir_ws[batch, 0], face_dir_ws[batch, 1], 0.0]) * scale
      visualizer.add_arrow(
        origin, origin + face_vec, color=(0.8, 0.0, 0.0, 0.7), width=0.015
      )

      # Actual linear velocity arrow (cyan) in world frame.
      vel_vec = np.array([lin_vel_ws[batch, 0], lin_vel_ws[batch, 1], 0.0]) * scale
      visualizer.add_arrow(
        origin, origin + vel_vec, color=(0.0, 0.6, 1.0, 0.7), width=0.015
      )


@dataclass(kw_only=True)
class SteeringCommandCfg(CommandTermCfg):
  entity_name: str
  rand_tar_dir: bool = True
  rand_face_dir: bool = True
  tar_speed_min: float = 0.5
  tar_speed_max: float = 3.0
  speed_deadzone: float = 0.0
  speed_deadzone_min: float | None = None
  speed_deadzone_max: float | None = None
  deadzone_sample_prob: float = 0.0
  stand_sample_prob: float = 0.0
  stand_speed: float = 0.0
  stand_speed_tolerance: float = 1e-4
  run_speed_min: float | None = None
  run_speed_max: float | None = None

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = None  # type: ignore[assignment]

  def __post_init__(self) -> None:
    if self.viz is None:
      self.viz = SteeringCommandCfg.VizCfg()

  def build(self, env: "ManagerBasedRlEnv") -> SteeringCommand:
    return SteeringCommand(self, env)


class BodyVelocityCommand(CommandTerm):
  """Periodic body-frame xy velocity and yaw-rate command."""

  cfg: "BodyVelocityCommandCfg"

  def __init__(self, cfg: "BodyVelocityCommandCfg", env: "ManagerBasedRlEnv"):
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
    heading_w = data.heading_w
    lin_vel_xy_b = _dir_world_to_local(data.root_link_lin_vel_w[:, :2], heading_w)
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

    stand_mask = torch.rand(n, device=self.device) < self.cfg.stand_sample_prob
    if self.cfg.reset_stand_mask_attr is not None:
      reset_labels = getattr(self._env, self.cfg.reset_stand_mask_attr, None)
      if reset_labels is not None:
        forced_mask = reset_labels[env_ids] >= 0
        if forced_mask.any():
          stand_mask[forced_mask] = reset_labels[env_ids[forced_mask]].bool()
          reset_labels[env_ids[forced_mask]] = -1

    lin_vel_b = torch.empty(n, 2, device=self.device)
    if stand_mask.any():
      lin_vel_b[stand_mask, 0] = torch.empty(
        int(stand_mask.sum()), device=self.device
      ).uniform_(self.cfg.stand_lin_vel_x_min, self.cfg.stand_lin_vel_x_max)
      lin_vel_b[stand_mask, 1] = torch.empty(
        int(stand_mask.sum()), device=self.device
      ).uniform_(self.cfg.stand_lin_vel_y_min, self.cfg.stand_lin_vel_y_max)
    if (~stand_mask).any():
      moving_count = int((~stand_mask).sum())
      lin_vel_b[~stand_mask, 0] = torch.empty(moving_count, device=self.device).uniform_(
        self.cfg.lin_vel_x_min, self.cfg.lin_vel_x_max
      )
      lin_vel_b[~stand_mask, 1] = torch.empty(moving_count, device=self.device).uniform_(
        self.cfg.lin_vel_y_min, self.cfg.lin_vel_y_max
      )
    self.lin_vel_b[env_ids] = lin_vel_b

    yaw_rate = torch.empty(n, device=self.device)
    if stand_mask.any():
      yaw_rate[stand_mask] = torch.empty(int(stand_mask.sum()), device=self.device).uniform_(
        -self.cfg.stand_yaw_rate_max, self.cfg.stand_yaw_rate_max
      )
    if (~stand_mask).any():
      yaw_rate[~stand_mask] = torch.empty(int((~stand_mask).sum()), device=self.device).uniform_(
        self.cfg.yaw_rate_min, self.cfg.yaw_rate_max
      )
    self.yaw_rate[env_ids] = yaw_rate
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

      # Body x velocity command (green), shown in world frame.
      x_vec = x_axis_w * lin_vel_bs[batch, 0] * scale
      visualizer.add_arrow(
        origin, origin + x_vec, color=(0.0, 0.75, 0.2, 0.75), width=0.015
      )

      # Actual body x velocity (light green), slightly higher than command arrows.
      actual_x_vec = x_axis_w * actual_lin_vel_bs[batch, 0] * scale
      visualizer.add_arrow(
        actual_origin,
        actual_origin + actual_x_vec,
        color=(0.55, 1.0, 0.55, 0.75),
        width=0.01,
      )

      # Body y velocity command (orange), shown in world frame.
      y_vec = y_axis_w * lin_vel_bs[batch, 1] * scale
      visualizer.add_arrow(
        origin, origin + y_vec, color=(1.0, 0.55, 0.0, 0.75), width=0.015
      )

      # Actual body y velocity (light orange).
      actual_y_vec = y_axis_w * actual_lin_vel_bs[batch, 1] * scale
      visualizer.add_arrow(
        actual_origin,
        actual_origin + actual_y_vec,
        color=(1.0, 0.85, 0.35, 0.75),
        width=0.01,
      )

      # Yaw-rate command (purple), drawn as a tangential arrow around the base.
      yaw_start = origin + x_axis_w * yaw_radius + np.array([0.0, 0.0, 0.12])
      yaw_vec = y_axis_w * yaw_rates[batch] * yaw_scale
      visualizer.add_arrow(
        yaw_start, yaw_start + yaw_vec, color=(0.55, 0.15, 1.0, 0.8), width=0.015
      )

      # Actual yaw rate (light purple), offset a little farther out.
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
  stand_sample_prob: float = 0.2
  stand_lin_vel_x_min: float = -0.15
  stand_lin_vel_x_max: float = 0.15
  stand_lin_vel_y_min: float = -0.15
  stand_lin_vel_y_max: float = 0.15
  stand_yaw_rate_max: float = 0.2
  reset_stand_mask_attr: str | None = None

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
    if self.stand_lin_vel_x_max < self.stand_lin_vel_x_min:
      msg = (
        f"stand_lin_vel_x_max ({self.stand_lin_vel_x_max}) must be >= "
        f"stand_lin_vel_x_min ({self.stand_lin_vel_x_min})."
      )
      raise ValueError(msg)
    if self.stand_lin_vel_y_max < self.stand_lin_vel_y_min:
      msg = (
        f"stand_lin_vel_y_max ({self.stand_lin_vel_y_max}) must be >= "
        f"stand_lin_vel_y_min ({self.stand_lin_vel_y_min})."
      )
      raise ValueError(msg)

  def build(self, env: "ManagerBasedRlEnv") -> BodyVelocityCommand:
    return BodyVelocityCommand(self, env)
