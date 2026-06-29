"""Roll out an SMP task and save the exact motion windows fed to the prior."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from tensordict import TensorDict

import smp.rl.tasks  # noqa: F401  # register Smp-* tasks
from smp.rl.events import NUM_JOINTS
from smp.rl.rewards import _update_buffer_from_sim
from smp.rl.utils import MotionFeatureBuffer
from smp.sampling.feature_to_state import EE_BODY_NAMES, NUM_EE


def _set_compile_model(env_cfg, compile_model: bool) -> None:
  init_event = env_cfg.events.get("init_smp_state")
  if init_event is not None and "compile_model" in init_event.params:
    init_event.params["compile_model"] = compile_model


def _set_gsi_size(env_cfg, buffer_size: int | None, batch_size: int | None) -> None:
  init_event = env_cfg.events.get("init_smp_state")
  if init_event is None:
    return
  if buffer_size is not None and "gsi_buffer_size" in init_event.params:
    init_event.params["gsi_buffer_size"] = buffer_size
  if batch_size is not None and "gsi_batch_size" in init_event.params:
      init_event.params["gsi_batch_size"] = batch_size


def _get_default_window_ckpt(env_cfg) -> str | None:
  init_event = env_cfg.events.get("init_smp_state")
  if init_event is None:
    return None
  return init_event.params.get("ckpt_path")


def _disable_smp_events_and_rewards(env_cfg) -> None:
  env_cfg.events.pop("init_smp_state", None)
  env_cfg.events.pop("gsi_reset", None)
  env_cfg.events.pop("gsi_refresh", None)
  env_cfg.rewards = {}


def _read_window_shape(ckpt_path: str) -> tuple[int, int]:
  ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
  cfg = ckpt["cfg"]
  return int(cfg["window_size"]), int(cfg["feature_dim"])


def _init_feature_buffer(raw_env: ManagerBasedRlEnv, window_size: int) -> None:
  robot = raw_env.scene["robot"]
  raw_env._smp_ee_indexes = torch.tensor(  # type: ignore[attr-defined]
    robot.find_bodies(list(EE_BODY_NAMES), preserve_order=True)[0],
    dtype=torch.long,
    device=raw_env.device,
  )
  raw_env._smp_buffer = MotionFeatureBuffer(  # type: ignore[attr-defined]
    num_envs=raw_env.num_envs,
    window_size=window_size,
    num_joints=NUM_JOINTS,
    num_ee=NUM_EE,
    device=raw_env.device,
  )
  for _ in range(window_size):
    _update_buffer_from_sim(raw_env)


def _set_fixed_command(raw_env: ManagerBasedRlEnv, command_name: str, command: list[float] | None) -> None:
  if command is None or command_name not in raw_env.command_manager.active_terms:
    return
  command_term = raw_env.command_manager.get_term(command_name)
  values = torch.tensor(command, device=raw_env.device, dtype=torch.float32)
  if values.numel() != 3:
    raise ValueError("--command expects exactly three values: vx vy yaw_rate")
  if hasattr(command_term, "lin_vel_b") and hasattr(command_term, "yaw_rate"):
    command_term.lin_vel_b[:] = values[:2]
    command_term.yaw_rate[:] = values[2]
    if hasattr(command_term, "is_heading_env"):
      command_term.is_heading_env[:] = False
    if hasattr(command_term, "is_standing_env"):
      command_term.is_standing_env[:] = False
    if hasattr(command_term, "command_b"):
      command_term.command_b[:, 0:2] = values[:2]
      command_term.command_b[:, 2] = values[2]
    return
  if hasattr(command_term, "command_b"):
    command_term.command_b[:] = values


def _compute_obs(env: RslRlVecEnvWrapper) -> TensorDict:
  obs_dict = env.unwrapped.observation_manager.compute(update_history=True)
  return TensorDict(obs_dict, batch_size=[env.num_envs])


def _make_policy(agent: str, env: RslRlVecEnvWrapper, task: str, checkpoint: str | None, device: str):
  if agent == "zero":
    action_shape = env.unwrapped.action_space.shape

    def policy(obs):
      del obs
      return torch.zeros(action_shape, device=env.unwrapped.device)

    return policy

  if agent == "random":
    action_shape = env.unwrapped.action_space.shape

    def policy(obs):
      del obs
      return 2.0 * torch.rand(action_shape, device=env.unwrapped.device) - 1.0

    return policy

  if checkpoint is None:
    raise ValueError("--checkpoint is required when --agent trained")
  ckpt_path = Path(checkpoint)
  if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

  agent_cfg = load_rl_cfg(task)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(ckpt_path), load_cfg={"actor": True}, strict=True, map_location=device)
  return runner.get_inference_policy(device=device)


def _get_command(raw_env: ManagerBasedRlEnv, command_name: str) -> np.ndarray | None:
  if command_name not in raw_env.command_manager.active_terms:
    return None
  command = raw_env.command_manager.get_command(command_name)
  if command is None:
    return None
  return command.detach().cpu().numpy()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("task", help="Registered task name, e.g. Smp-BodyVelocity-G1")
  parser.add_argument("--checkpoint", help="RSL-RL checkpoint for --agent trained")
  parser.add_argument("--agent", choices=("trained", "zero", "random"), default="trained")
  parser.add_argument("--output", required=True, help="Output .npz path")
  parser.add_argument("--device", default=None, help="Device, default cuda:0 if available else cpu")
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--steps", type=int, default=400)
  parser.add_argument("--warmup-steps", type=int, default=0)
  parser.add_argument("--stride", type=int, default=1, help="Save every N rollout steps")
  parser.add_argument("--record-envs", type=int, default=None, help="Number of envs to save")
  parser.add_argument("--command-name", default="steering")
  parser.add_argument("--command", nargs=3, type=float, default=None, metavar=("VX", "VY", "YAW_RATE"))
  parser.add_argument("--no-terminations", action="store_true")
  parser.add_argument("--use-gsi", action="store_true", help="Use the task's original SMP/GSI reset path")
  parser.add_argument("--window-ckpt", default=None, help="Diffusion checkpoint used only to infer window shape")
  parser.add_argument("--compile-model", action="store_true", help="Only relevant with --use-gsi")
  parser.add_argument("--gsi-buffer-size", type=int, default=None)
  parser.add_argument("--gsi-batch-size", type=int, default=None)
  parser.add_argument("--seed", type=int, default=None)
  args = parser.parse_args()

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(args.task, play=True)
  default_window_ckpt = _get_default_window_ckpt(env_cfg)
  env_cfg.scene.num_envs = args.num_envs
  if args.seed is not None:
    env_cfg.seed = args.seed
  if args.no_terminations:
    env_cfg.terminations = {}
  if args.use_gsi:
    _set_compile_model(env_cfg, args.compile_model)
    _set_gsi_size(env_cfg, args.gsi_buffer_size, args.gsi_batch_size)
  else:
    _disable_smp_events_and_rewards(env_cfg)

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  agent_cfg = load_rl_cfg(args.task)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  if not args.use_gsi:
    window_ckpt = args.window_ckpt or default_window_ckpt
    if window_ckpt is None:
      raise ValueError("--window-ckpt is required when the task has no init_smp_state event")
    window_size, feature_dim = _read_window_shape(window_ckpt)
    _init_feature_buffer(raw_env, window_size)
  else:
    feature_dim = int(raw_env._smp_buffer.compute_features().shape[-1])  # type: ignore[attr-defined]
  _set_fixed_command(raw_env, args.command_name, args.command)
  policy = _make_policy(args.agent, env, args.task, args.checkpoint, device)

  record_envs = args.record_envs or args.num_envs
  record_envs = min(record_envs, args.num_envs)
  env_ids = torch.arange(record_envs, device=raw_env.device)

  windows: list[np.ndarray] = []
  steps: list[np.ndarray] = []
  env_id_rows: list[np.ndarray] = []
  commands: list[np.ndarray] = []
  rewards: list[np.ndarray] = []
  dones: list[np.ndarray] = []

  obs = _compute_obs(env)
  with torch.inference_mode():
    total_steps = args.warmup_steps + args.steps
    for step in range(total_steps):
      action = policy(obs)
      obs, reward, done, _extras = env.step(action)
      _set_fixed_command(raw_env, args.command_name, args.command)
      _update_buffer_from_sim(raw_env)
      obs = _compute_obs(env)
      if step < args.warmup_steps:
        continue
      rollout_step = step - args.warmup_steps
      if rollout_step % args.stride != 0:
        continue

      features = raw_env._smp_buffer.compute_features()[env_ids]  # type: ignore[attr-defined]
      windows.append(features.detach().cpu().numpy())
      steps.append(np.full((record_envs,), rollout_step, dtype=np.int64))
      env_id_rows.append(np.arange(record_envs, dtype=np.int64))
      rewards.append(reward[env_ids].detach().cpu().numpy())
      dones.append(done[env_ids].detach().cpu().numpy())

      command = _get_command(raw_env, args.command_name)
      if command is not None:
        commands.append(command[:record_envs])

  env.close()

  if not windows:
    raise RuntimeError("No windows were collected. Increase --steps or reduce --warmup-steps.")

  windows_arr = np.concatenate(windows, axis=0)
  step_arr = np.concatenate(steps, axis=0)
  env_id_arr = np.concatenate(env_id_rows, axis=0)
  reward_arr = np.concatenate(rewards, axis=0)
  done_arr = np.concatenate(dones, axis=0)
  command_arr = np.concatenate(commands, axis=0) if commands else np.empty((0,), dtype=np.float32)

  metadata = {
    "task": args.task,
    "checkpoint": args.checkpoint,
    "agent": args.agent,
    "device": device,
    "num_envs": args.num_envs,
    "record_envs": record_envs,
    "steps": args.steps,
    "warmup_steps": args.warmup_steps,
    "stride": args.stride,
    "command_name": args.command_name,
    "command": args.command,
    "use_gsi": args.use_gsi,
    "window_ckpt": args.window_ckpt or default_window_ckpt,
    "window_size": int(windows_arr.shape[1]),
    "feature_dim": int(feature_dim),
  }

  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    output,
    windows=windows_arr.astype(np.float32, copy=False),
    step=step_arr,
    env_id=env_id_arr,
    command=command_arr.astype(np.float32, copy=False),
    reward=reward_arr.astype(np.float32, copy=False),
    done=done_arr,
    metadata=json.dumps(metadata),
  )
  print(f"[INFO] Saved {windows_arr.shape[0]} windows to {output}")
  print(f"[INFO] windows shape: {windows_arr.shape}")


if __name__ == "__main__":
  main()
