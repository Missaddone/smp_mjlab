#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_smp_onnx}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/export_onnx.sh <TASK_ID> <CHECKPOINT_FILE>

Example:
  bash scripts/export_onnx.sh \
    Smp-BodyVelocity-Exp13-Group7-G1 \
    logs/rsl_rl/smp_exp13_body_velocity_moving_reward_mix/wandb_checkpoints/04e1a64a/model_12998.pt

The output is written next to the checkpoint as <checkpoint_stem>.onnx.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -ne 2 ]]; then
  usage >&2
  exit 2
fi

TASK_ID="$1"
CHECKPOINT_FILE="$2"

if [[ ! -f "$CHECKPOINT_FILE" ]]; then
  echo "[ERROR] Checkpoint not found: $CHECKPOINT_FILE" >&2
  exit 1
fi

echo "[INFO] Exporting $TASK_ID"
echo "[INFO] Checkpoint: $CHECKPOINT_FILE"

uv run python - "$TASK_ID" "$CHECKPOINT_FILE" <<'PY'
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

import smp.rl.tasks  # noqa: F401  # register Smp-* tasks


def main() -> None:
  task_id = sys.argv[1]
  checkpoint = Path(sys.argv[2]).resolve()
  output = checkpoint.with_suffix(".onnx")
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
    runner.export_policy_to_onnx(str(checkpoint.parent), filename=output.name)
  finally:
    wrapped_env.close()

  print(f"[DONE] ONNX exported: {output}")


if __name__ == "__main__":
  main()
PY
