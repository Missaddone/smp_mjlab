"""Goalkeeper command: face incoming shots and protect the goal behind."""

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


def _xy_world_to_local(vec_w: torch.Tensor, heading_w: torch.Tensor) -> torch.Tensor:
  """Rotate a (N, 2) world-frame xy vector into the heading-aligned frame."""
  cos_h = torch.cos(heading_w)
  sin_h = torch.sin(heading_w)
  x_w, y_w = vec_w[..., 0], vec_w[..., 1]
  return torch.stack([cos_h * x_w + sin_h * y_w, -sin_h * x_w + cos_h * y_w], dim=-1)


class CatchBallCommand(CommandTerm):
  """Incoming shot state plus required goalkeeper facing direction.

  The policy observes ``[face_dir_b(2), ball_rel_pos_b(3), ball_lin_vel_b(3)]``.
  The ball is a free-floating MuJoCo body. Every resample period it is launched
  from a randomized shooter position in front of the robot toward the goal
  behind the robot.
  """

  cfg: CatchBallCommandCfg

  def __init__(self, cfg: CatchBallCommandCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.ball: Entity = env.scene[cfg.ball_entity_name]
    self.goal: Entity = env.scene[cfg.goal_entity_name]

    self.face_dir_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.launch_dir_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.launch_speed = torch.zeros(self.num_envs, device=self.device)
    self.ball_age = torch.zeros(self.num_envs, device=self.device)
    self.ball_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    self.episode_blocks = torch.zeros(self.num_envs, device=self.device)
    self.episode_goals_allowed = torch.zeros(self.num_envs, device=self.device)
    self.goal_dir_w = self._unit_xy(self.cfg.goal_direction, self.num_envs)
    self.goal_left_w = torch.stack([-self.goal_dir_w[:, 1], self.goal_dir_w[:, 0]], dim=-1)
    self.face_dir_w[:] = -self.goal_dir_w

    self.command_b = torch.zeros(self.num_envs, 8, device=self.device)

    self.metrics["ball_distance"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_face"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["episode_blocks"] = self.episode_blocks
    self.metrics["episode_goals_allowed"] = self.episode_goals_allowed

    self._gui_enabled: viser.GuiCheckboxHandle | None = None
    self._gui_angle: viser.GuiSliderHandle | None = None
    self._gui_speed: viser.GuiSliderHandle | None = None
    self._gui_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.command_b

  def _unit_xy(self, xy: tuple[float, float], n: int) -> torch.Tensor:
    vec = torch.tensor(xy, device=self.device, dtype=torch.float32)
    vec = vec / torch.clamp(torch.norm(vec), min=1.0e-6)
    return vec.unsqueeze(0).repeat(n, 1)

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    self._update_goal_frame(env_ids)
    extras = super().reset(env_ids)
    self.ball_active[env_ids] = True
    self._place_goal(env_ids)
    return extras

  def _update_goal_frame(self, env_ids: torch.Tensor) -> None:
    if self.cfg.align_to_robot_heading:
      heading_w = self.robot.data.heading_w[env_ids]
      face_dir = torch.stack([torch.cos(heading_w), torch.sin(heading_w)], dim=-1)
      self.face_dir_w[env_ids] = face_dir
      self.goal_dir_w[env_ids] = -face_dir
    else:
      self.goal_dir_w[env_ids] = self._unit_xy(self.cfg.goal_direction, len(env_ids))
      self.face_dir_w[env_ids] = -self.goal_dir_w[env_ids]
    self.goal_left_w[env_ids] = torch.stack(
      [-self.goal_dir_w[env_ids, 1], self.goal_dir_w[env_ids, 0]], dim=-1
    )

  def goal_center_w(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
    origins = self._env.scene.env_origins[env_ids]
    return origins + torch.cat(
      [
        self.goal_dir_w[env_ids] * self.cfg.goal_line_distance,
        torch.zeros((len(env_ids), 1), device=self.device),
      ],
      dim=-1,
    )

  def _place_goal(self, env_ids: torch.Tensor) -> None:
    goal_pose = torch.zeros((len(env_ids), 7), device=self.device)
    goal_pose[:, :3] = self.goal_center_w(env_ids)
    goal_pose[:, 3] = 1.0
    self.goal.write_mocap_pose_to_sim(goal_pose, env_ids)

  def ball_link_distance(self) -> torch.Tensor:
    link_pos = self.robot.data.body_link_pos_w
    return torch.norm(link_pos - self.ball_pos_w[:, None, :], dim=-1).amin(dim=-1)

  def ball_contact(self, catch_distance: float) -> torch.Tensor:
    sensors = getattr(self._env.scene, "sensors", {})
    sensor = sensors.get(self.cfg.contact_sensor_name)
    if sensor is not None:
      found = sensor.data.found
      if found.ndim > 1:
        found = found.any(dim=tuple(range(1, found.ndim)))
      return found.bool()
    return self.ball_link_distance() <= catch_distance

  def ball_crossed_goal_line(self) -> torch.Tensor:
    goal_center = self.goal_center_w()[:, :2]
    ball_rel = self.ball_pos_w[:, :2] - goal_center
    along_goal = (ball_rel * self.goal_dir_w).sum(dim=-1)
    lateral_goal = (ball_rel * self.goal_left_w).sum(dim=-1).abs()
    ball_height = self.ball_pos_w[:, 2]
    return (along_goal >= 0.0) & (
      lateral_goal <= self.cfg.goal_width * 0.5
    ) & (ball_height <= self.cfg.goal_height)

  def ball_expired(self) -> torch.Tensor:
    return self.ball_age >= self.cfg.max_ball_lifetime

  def block_contact(self, catch_distance: float | None = None) -> torch.Tensor:
    if catch_distance is None:
      catch_distance = self.cfg.catch_metric_distance
    return self.ball_active & self.ball_contact(catch_distance)

  @property
  def ball_pos_w(self) -> torch.Tensor:
    return self.ball.data.root_link_pos_w

  @property
  def ball_lin_vel_w(self) -> torch.Tensor:
    return self.ball.data.root_link_lin_vel_w

  def _update_metrics(self) -> None:
    max_step = self.cfg.resampling_time_range[1] / self._env.step_dt
    root_to_ball = self.ball_pos_w - self.robot.data.root_link_pos_w
    self.metrics["ball_distance"] += torch.norm(root_to_ball, dim=-1) / max_step

    heading_w = self.robot.data.heading_w
    char_face_w = torch.stack([torch.cos(heading_w), torch.sin(heading_w)], dim=-1)
    face_dot = (self.face_dir_w * char_face_w).sum(dim=-1)
    self.metrics["error_face"] += (1.0 - face_dot.clamp(-1.0, 1.0)) / max_step

    blocked = self.block_contact()
    scored = self.ball_active & self.ball_crossed_goal_line()
    expired = self.ball_active & self.ball_expired()

    self.episode_blocks += blocked.float()
    self.episode_goals_allowed += scored.float()

    done = blocked | scored | expired
    if done.any():
      self._hide_ball(done.nonzero().flatten())

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = int(env_ids.numel())
    self._update_goal_frame(env_ids)
    goal_center = self.goal_center_w(env_ids)

    self._place_goal(env_ids)

    goal_theta = torch.atan2(self.goal_dir_w[env_ids, 1], self.goal_dir_w[env_ids, 0])
    launch_theta = goal_theta + torch.empty(n, device=self.device).uniform_(
      -self.cfg.launch_angle_jitter, self.cfg.launch_angle_jitter
    )
    launch_dir = torch.stack(
      [torch.cos(launch_theta), torch.sin(launch_theta)], dim=-1
    )
    speed = torch.empty(n, device=self.device).uniform_(
      self.cfg.ball_speed_min, self.cfg.ball_speed_max
    )
    spawn_distance = torch.empty(n, device=self.device).uniform_(
      self.cfg.spawn_distance_min, self.cfg.spawn_distance_max
    )
    spawn_height = torch.empty(n, device=self.device).uniform_(
      self.cfg.spawn_height_min, self.cfg.spawn_height_max
    )
    lateral = torch.empty(n, device=self.device).uniform_(
      -self.cfg.shooter_lateral_range, self.cfg.shooter_lateral_range
    )
    ball_pos = torch.zeros(n, 3, device=self.device)
    ball_pos[:, :2] = goal_center[:, :2] - self.goal_dir_w[
      env_ids
    ] * spawn_distance.unsqueeze(-1) + self.goal_left_w[env_ids] * lateral.unsqueeze(-1)
    ball_pos[:, 2] = spawn_height

    vertical_vel = torch.empty(n, device=self.device).uniform_(
      self.cfg.ball_vertical_velocity_min, self.cfg.ball_vertical_velocity_max
    )
    ball_vel = torch.zeros(n, 3, device=self.device)
    ball_vel[:, :2] = launch_dir * speed.unsqueeze(-1)
    ball_vel[:, 2] = vertical_vel

    root_state = torch.zeros(n, 13, device=self.device)
    root_state[:, :3] = ball_pos
    root_state[:, 3] = 1.0
    root_state[:, 7:10] = ball_vel
    self.ball.write_root_state_to_sim(root_state, env_ids)

    self.launch_dir_w[env_ids] = launch_dir
    self.launch_speed[env_ids] = speed
    self.ball_age[env_ids] = 0.0
    self.ball_active[env_ids] = True

  def _hide_ball(self, env_ids: torch.Tensor) -> None:
    n = int(env_ids.numel())
    root_state = torch.zeros(n, 13, device=self.device)
    root_state[:, 0:2] = self.goal_center_w(env_ids)[:, :2]
    root_state[:, 2] = self.cfg.inactive_ball_height
    root_state[:, 3] = 1.0
    self.ball.write_root_state_to_sim(root_state, env_ids)
    self.ball_active[env_ids] = False

  def _update_command(self) -> None:
    dt = self._env.step_dt
    self.ball_age += dt
    self._refresh_command()

  def _refresh_command(self) -> None:
    heading_w = self.robot.data.heading_w
    root_pos = self.robot.data.root_link_pos_w
    rel_pos_w = self.ball_pos_w - root_pos

    self.command_b[:, 0:2] = _xy_world_to_local(self.face_dir_w, heading_w)
    self.command_b[:, 2:4] = _xy_world_to_local(rel_pos_w[:, :2], heading_w)
    self.command_b[:, 4] = rel_pos_w[:, 2]
    self.command_b[:, 5:7] = _xy_world_to_local(self.ball_lin_vel_w[:, :2], heading_w)
    self.command_b[:, 7] = self.ball_lin_vel_w[:, 2]

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create manual catch-ball controls in the Viser viewer."""
    from viser import Icon

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)
      angle_slider = server.gui.add_slider(
        "face_angle (rad)",
        min=-math.pi,
        max=math.pi,
        step=0.05,
        initial_value=0.0,
      )
      speed_slider = server.gui.add_slider(
        "ball_speed",
        min=float(self.cfg.ball_speed_min),
        max=float(self.cfg.ball_speed_max),
        step=0.1,
        initial_value=float((self.cfg.ball_speed_min + self.cfg.ball_speed_max) * 0.5),
      )
      fire_btn = server.gui.add_button("Fire ball", icon=Icon.PLAY)

      @fire_btn.on_click
      def _(_) -> None:
        idx = get_env_idx()
        self._spawn_gui_ball(idx)
        self._refresh_command()

    self._gui_enabled = enabled
    self._gui_angle = angle_slider
    self._gui_speed = speed_slider
    self._gui_get_env_idx = get_env_idx

  def _spawn_gui_ball(self, idx: int) -> None:
    assert self._gui_angle is not None
    assert self._gui_speed is not None
    angle = float(self._gui_angle.value)
    speed = float(self._gui_speed.value)
    launch_dir = torch.tensor(
      [math.cos(angle), math.sin(angle)], device=self.device, dtype=torch.float32
    )
    env_ids = torch.tensor([idx], device=self.device, dtype=torch.long)
    goal_center = self.goal_center_w(env_ids)[0]
    self.face_dir_w[idx] = -self.goal_dir_w[idx]
    self._place_goal(env_ids)
    ball_pos = torch.zeros(3, device=self.device)
    ball_pos[:2] = goal_center[:2] - self.goal_dir_w[idx] * self.cfg.spawn_distance_max
    ball_pos[2] = self.cfg.spawn_height_max
    ball_vel = torch.zeros(3, device=self.device)
    ball_vel[:2] = launch_dir * speed
    root_state = torch.zeros(1, 13, device=self.device)
    root_state[0, :3] = ball_pos
    root_state[0, 3] = 1.0
    root_state[0, 7:10] = ball_vel
    self.ball.write_root_state_to_sim(root_state, env_ids)
    self.launch_dir_w[idx] = launch_dir
    self.launch_speed[idx] = speed
    self.ball_age[idx] = 0.0
    self.ball_active[idx] = True

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._gui_enabled is None or not self._gui_enabled.value:
      return
    assert self._gui_get_env_idx is not None
    assert self._gui_angle is not None
    idx = self._gui_get_env_idx()
    self.face_dir_w[idx] = -self.goal_dir_w[idx]
    self._refresh_command()

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    root_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    face_dir_ws = self.face_dir_w.cpu().numpy()
    ball_pos_ws = self.ball_pos_w.cpu().numpy()
    ball_vel_ws = self.ball_lin_vel_w.cpu().numpy()

    radius = float(self.cfg.ball_radius)
    scale = float(self.cfg.viz.velocity_scale)

    for batch in env_indices:
      root_pos = root_pos_ws[batch]
      if np.linalg.norm(root_pos) < 1e-6:
        continue

      ball_pos = ball_pos_ws[batch]
      ball_vel = ball_vel_ws[batch] * scale
      face_vec = np.array([face_dir_ws[batch, 0], face_dir_ws[batch, 1], 0.0])

      visualizer.add_sphere(ball_pos, radius=radius, color=(0.95, 0.72, 0.18, 0.9))
      visualizer.add_arrow(
        ball_pos,
        ball_pos + ball_vel,
        color=(0.95, 0.72, 0.18, 0.65),
        width=0.012,
      )
      visualizer.add_arrow(
        root_pos + np.array([0.0, 0.0, self.cfg.viz.face_z_offset]),
        root_pos + np.array([0.0, 0.0, self.cfg.viz.face_z_offset]) + face_vec,
        color=(0.8, 0.0, 0.0, 0.75),
        width=0.015,
      )
      goal_center = self.goal_center_w(
        torch.tensor([batch], device=self.device, dtype=torch.long)
      )[0].cpu().numpy()
      goal_line_center = goal_center + np.array([0.0, 0.0, 0.03])
      goal_half = np.array(
        [
          self.goal_left_w[batch, 0].item() * self.cfg.goal_width * 0.5,
          self.goal_left_w[batch, 1].item() * self.cfg.goal_width * 0.5,
          0.0,
        ]
      )
      visualizer.add_cylinder(
        goal_line_center - goal_half,
        goal_line_center + goal_half,
        radius=0.015,
        color=(0.1, 0.8, 0.2, 0.75),
      )
      keeper_line_center = goal_center + np.array(
        [
          -self.goal_dir_w[batch, 0].item() * self.cfg.viz.keeper_line_distance,
          -self.goal_dir_w[batch, 1].item() * self.cfg.viz.keeper_line_distance,
          0.03,
        ]
      )
      visualizer.add_cylinder(
        keeper_line_center - goal_half,
        keeper_line_center + goal_half,
        radius=0.012,
        color=(0.1, 0.35, 1.0, 0.75),
      )


@dataclass(kw_only=True)
class CatchBallCommandCfg(CommandTermCfg):
  entity_name: str
  ball_entity_name: str = "football"
  goal_entity_name: str = "goal"
  contact_sensor_name: str = "football_contact"
  align_to_robot_heading: bool = True
  goal_direction: tuple[float, float] = (-1.0, 0.0)
  spawn_distance_min: float = 3.0
  spawn_distance_max: float = 5.0
  spawn_height_min: float = 0.7
  spawn_height_max: float = 1.4
  shooter_lateral_range: float = 1.2
  launch_angle_jitter: float = 0.15
  goal_line_distance: float = 0.8
  goal_width: float = 2.4
  goal_height: float = 1.6
  max_ball_lifetime: float = 4.5
  inactive_ball_height: float = -10.0
  ball_radius: float = 0.11
  catch_metric_distance: float = 0.22
  ball_speed_min: float = 4.0
  ball_speed_max: float = 8.0
  ball_vertical_velocity_min: float = -0.2
  ball_vertical_velocity_max: float = 0.3

  @dataclass
  class VizCfg:
    face_z_offset: float = 0.2
    velocity_scale: float = 0.35
    keeper_line_distance: float = 0.5

  viz: VizCfg = None  # type: ignore[assignment]

  def __post_init__(self) -> None:
    if self.viz is None:
      self.viz = CatchBallCommandCfg.VizCfg()
    if self.ball_speed_max < self.ball_speed_min:
      msg = (
        f"ball_speed_max ({self.ball_speed_max}) must be >= "
        f"ball_speed_min ({self.ball_speed_min})."
      )
      raise ValueError(msg)

  def build(self, env: "ManagerBasedRlEnv") -> CatchBallCommand:
    return CatchBallCommand(self, env)
