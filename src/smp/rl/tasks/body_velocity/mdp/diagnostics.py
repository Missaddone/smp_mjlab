"""Experiment 15 diagnostics and file-backed rollout recorder."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from mjlab.managers.recorder_manager import RecorderTerm

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DIAGNOSTIC_NAMES = (
  "action_delta_rms",
  "action_delta_peak",
  "action_accel_proxy_rms",
  "action_accel_proxy_peak",
  "joint_vel_rms",
  "joint_vel_peak",
  "leg_action_delta_rms",
  "leg_action_delta_peak",
  "leg_action_accel_proxy_rms",
  "leg_action_accel_proxy_peak",
  "leg_action_sign_flip_rate",
  "leg_joint_vel_rms",
  "leg_joint_vel_peak",
  "foot_action_delta_rms",
  "foot_action_delta_peak",
  "foot_action_accel_proxy_rms",
  "foot_action_accel_proxy_peak",
  "foot_action_sign_flip_rate",
  "foot_joint_vel_rms",
  "foot_joint_vel_peak",
)

_SIGN_FLIP_MIN_DELTA = 1.0e-3


def action_diagnostic_values(
  action: torch.Tensor,
  previous_action: torch.Tensor,
  previous_previous_action: torch.Tensor,
  joint_vel: torch.Tensor,
  action_groups: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
  """Return per-environment action and joint-velocity diagnostic values."""
  action_delta = action - previous_action
  previous_delta = previous_action - previous_previous_action
  action_accel_proxy = action_delta - previous_delta
  values = {
    "action_delta_rms": torch.sqrt(torch.mean(torch.square(action_delta), dim=-1)),
    "action_delta_peak": torch.amax(torch.abs(action_delta), dim=-1),
    "action_accel_proxy_rms": torch.sqrt(
      torch.mean(torch.square(action_accel_proxy), dim=-1)
    ),
    "action_accel_proxy_peak": torch.amax(torch.abs(action_accel_proxy), dim=-1),
    "joint_vel_rms": torch.sqrt(torch.mean(torch.square(joint_vel), dim=-1)),
    "joint_vel_peak": torch.amax(torch.abs(joint_vel), dim=-1),
  }
  for group_name, indices in (action_groups or {}).items():
    group_delta = torch.index_select(action_delta, dim=-1, index=indices)
    group_previous_delta = torch.index_select(previous_delta, dim=-1, index=indices)
    group_accel = group_delta - group_previous_delta
    group_joint_vel = torch.index_select(joint_vel, dim=-1, index=indices)
    sign_flip = (
      (group_delta * group_previous_delta < 0.0)
      & (torch.abs(group_delta) >= _SIGN_FLIP_MIN_DELTA)
      & (torch.abs(group_previous_delta) >= _SIGN_FLIP_MIN_DELTA)
    )
    values.update(
      {
        f"{group_name}_action_delta_rms": torch.sqrt(
          torch.mean(torch.square(group_delta), dim=-1)
        ),
        f"{group_name}_action_delta_peak": torch.amax(
          torch.abs(group_delta), dim=-1
        ),
        f"{group_name}_action_accel_proxy_rms": torch.sqrt(
          torch.mean(torch.square(group_accel), dim=-1)
        ),
        f"{group_name}_action_accel_proxy_peak": torch.amax(
          torch.abs(group_accel), dim=-1
        ),
        f"{group_name}_action_sign_flip_rate": torch.mean(
          sign_flip.to(action.dtype), dim=-1
        ),
        f"{group_name}_joint_vel_rms": torch.sqrt(
          torch.mean(torch.square(group_joint_vel), dim=-1)
        ),
        f"{group_name}_joint_vel_peak": torch.amax(
          torch.abs(group_joint_vel), dim=-1
        ),
      }
    )
  return values


def _action_group_indices(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
  cached = getattr(env, "_exp15_action_group_indices", None)
  if cached is not None:
    return cached

  action_term = env.action_manager.get_term("joint_pos")
  target_names = action_term.target_names
  leg_indices = [
    index
    for index, name in enumerate(target_names)
    if name.startswith(("left_", "right_"))
    and any(part in name for part in ("hip", "knee", "ankle"))
  ]
  foot_indices = [
    index for index, name in enumerate(target_names) if "ankle" in name
  ]
  if not leg_indices or not foot_indices:
    raise RuntimeError(
      "Exp15 diagnostics could not resolve leg/foot policy action indices from "
      f"joint_pos targets: {target_names}"
    )

  cached = {
    "leg": torch.tensor(leg_indices, device=env.device, dtype=torch.long),
    "foot": torch.tensor(foot_indices, device=env.device, dtype=torch.long),
  }
  setattr(env, "_exp15_action_group_indices", cached)
  return cached


def env_action_diagnostic_values(
  env: ManagerBasedRlEnv,
) -> dict[str, torch.Tensor]:
  """Compute Exp15 diagnostics once per policy step and reuse the result."""
  step = env.common_step_counter
  cached = getattr(env, "_exp15_action_diagnostics_cache", None)
  if cached is not None and cached[0] == step:
    return cached[1]

  values = action_diagnostic_values(
    env.action_manager.action,
    env.action_manager.prev_action,
    env.action_manager.prev_prev_action,
    _joint_velocity(env),
    action_groups=_action_group_indices(env),
  )
  setattr(env, "_exp15_action_diagnostics_cache", (step, values))
  return values


def _joint_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
  asset = env.scene["robot"]
  action_term = env.action_manager.get_term("joint_pos")
  return torch.index_select(asset.data.joint_vel, dim=-1, index=action_term.target_ids)


class Exp15DiagnosticsRecorder(RecorderTerm):
  """Write aggregate action/joint diagnostics to a CSV file during rollouts."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    path = Path(cfg.params["path"]).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    self._file = path.open("w", newline="", encoding="utf-8")
    self._writer = csv.writer(self._file)
    print(f"[Exp15 diagnostics] writing CSV: {path}", flush=True)
    self._writer.writerow(
      [
        "step",
        "active_envs",
        "reset_envs",
        "action_delta_rms",
        "action_delta_peak",
        "action_accel_proxy_rms",
        "action_accel_proxy_peak",
        "joint_vel_rms",
        "joint_vel_peak",
      ]
    )
    self._flush_interval = max(1, int(cfg.params.get("flush_interval_steps", 50)))
    self._step = 0
    self._reset_envs = 0

  def record_pre_reset(self, env_ids: torch.Tensor) -> None:
    self._reset_envs += int(env_ids.numel())

  def record_post_step(self) -> None:
    values = env_action_diagnostic_values(self._env)
    active = ~self._env.reset_buf
    active_count = int(active.sum().item())
    if active_count:
      means = {
        name: float(values[name][active].mean().detach().cpu().item())
        for name in _DIAGNOSTIC_NAMES
      }
    else:
      means = {name: 0.0 for name in _DIAGNOSTIC_NAMES}

    self._step += 1
    self._writer.writerow(
      [
        self._step,
        active_count,
        self._reset_envs,
        *(means[name] for name in _DIAGNOSTIC_NAMES),
      ]
    )
    self._reset_envs = 0
    if self._step % self._flush_interval == 0:
      self._file.flush()

  def close(self) -> None:
    self._file.flush()
    self._file.close()
