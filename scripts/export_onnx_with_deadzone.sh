#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_smp_onnx}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/export_onnx_with_deadzone.sh <TASK_ID> <CHECKPOINT_FILE>

Example:
  bash scripts/export_onnx_with_deadzone.sh \
    Smp-BodyVelocity-Exp10-Group19-G1 \
    logs/rsl_rl/smp_exp10_body_velocity_static_switch/<run>/model_9999.pt

The output is written next to the checkpoint as:
  <checkpoint_stem>_deadzone.onnx

Environment variables:
  DEAD_ZONE  Body-velocity command dead-zone threshold. Default: 0.5.
EOF
}

export_one() {
  local task_id="$1"
  local checkpoint_file="$2"
  local dead_zone="${DEAD_ZONE:-0.5}"

  if [[ ! -f "$checkpoint_file" ]]; then
    echo "[ERROR] Checkpoint not found: $checkpoint_file" >&2
    return 1
  fi

  echo "[INFO] Exporting body-velocity dead-zone ONNX for $task_id"
  echo "[INFO] Checkpoint: $checkpoint_file"
  echo "[INFO] Dead zone: $dead_zone"

  uv run python - "$task_id" "$checkpoint_file" "$dead_zone" <<'PY'
from __future__ import annotations

import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

import smp.rl.tasks  # noqa: F401  # register Smp-* tasks


class BodyVelocityDeadZoneOnnx(nn.Module):
  """Apply [x, y, yaw] command dead-zone before the exported actor."""

  def __init__(
    self,
    actor_onnx: nn.Module,
    command_start: int,
    dead_zone: float,
  ) -> None:
    super().__init__()
    self.actor_onnx = actor_onnx
    self.command_start = command_start
    self.command_dim = 3
    self.dead_zone = dead_zone
    self.input_size = actor_onnx.input_size  # type: ignore[attr-defined]

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    s = self.command_start
    e = s + self.command_dim
    command = obs[..., s:e]
    stop_mask = torch.linalg.norm(command, dim=-1) < self.dead_zone
    command = torch.where(stop_mask.unsqueeze(-1), torch.zeros_like(command), command)
    obs = torch.cat([obs[..., :s], command, obs[..., e:]], dim=-1)
    return self.actor_onnx(obs)

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]


def find_body_velocity_command_slice(env: ManagerBasedRlEnv) -> tuple[int, int]:
  obs_manager = env.observation_manager
  group_name = "actor"
  if not obs_manager.group_obs_concatenate[group_name]:
    raise RuntimeError("actor observation group must be concatenated for ONNX export")

  names = obs_manager.active_terms[group_name]
  dims = obs_manager.group_obs_term_dim[group_name]

  offset = 0
  for name, shape in zip(names, dims, strict=False):
    width = int(math.prod(shape))
    if name == "command":
      if width != 3:
        raise RuntimeError(
          "body-velocity deployment export expects a 3D [x, y, yaw] command, "
          f"but task actor command dim is {width}"
        )
      return offset, width
    offset += width

  raise RuntimeError(f"Cannot find 'command' in actor observation terms: {names}")


def main() -> None:
  task_id = sys.argv[1]
  checkpoint = Path(sys.argv[2]).resolve()
  dead_zone = float(sys.argv[3])
  output = checkpoint.with_name(f"{checkpoint.stem}_deadzone.onnx")

  if dead_zone < 0.0:
    raise ValueError(f"dead-zone must be non-negative, got {dead_zone}")
  if not checkpoint.exists():
    raise FileNotFoundError(checkpoint)

  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = 1

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  try:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped_env, asdict(agent_cfg), device=device)
    runner.load(
      str(checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )

    command_start, command_dim = find_body_velocity_command_slice(env)
    actor_onnx = runner.alg.get_policy().as_onnx(verbose=False)
    onnx_model = BodyVelocityDeadZoneOnnx(
      actor_onnx=actor_onnx,
      command_start=command_start,
      dead_zone=dead_zone,
    )
    onnx_model.to("cpu")
    onnx_model.eval()

    os.makedirs(output.parent, exist_ok=True)
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),
      str(output),
      export_params=True,
      opset_version=18,
      verbose=False,
      input_names=onnx_model.input_names,
      output_names=onnx_model.output_names,
      dynamic_axes={},
      dynamo=False,
    )

    metadata = get_base_metadata(env, str(checkpoint))
    metadata.update(
      {
        "export_type": "body_velocity_deadzone_actor_wrapper",
        "checkpoint_file": str(checkpoint),
        "deadzone_threshold": dead_zone,
        "command_start": command_start,
        "command_dim": command_dim,
      }
    )
    attach_metadata_to_onnx(str(output), metadata)
  finally:
    wrapped_env.close()

  print(f"[DONE] Dead-zone ONNX exported: {output}")


if __name__ == "__main__":
  main()
PY
}

case "${1:-}" in
  -h|--help)
    usage
    ;;
  *)
    if [[ "$#" -ne 2 ]]; then
      usage >&2
      exit 2
    fi
    export_one "$1" "$2"
    ;;
esac
