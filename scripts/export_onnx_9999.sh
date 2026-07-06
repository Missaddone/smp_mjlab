#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_smp_onnx}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/export_onnx_9999.sh <TASK_ID> <CHECKPOINT_FILE>
  bash scripts/export_onnx_9999.sh --all-exp4

Examples:
  bash scripts/export_onnx_9999.sh \
    Smp-BodyVelocity-Exp4-Group1-G1 \
    logs/rsl_rl/smp_exp4_body_velocity_ablation/2026-07-03_11-14-19_group1_mix_p07_l015_y015_kxy1_kyaw1/model_9999.pt

  bash scripts/export_onnx_9999.sh --all-exp4

The output is written next to model_9999.pt as model_9999.onnx.
EOF
}

export_one() {
  local task_id="$1"
  local checkpoint_file="$2"

  if [[ "$(basename "$checkpoint_file")" != "model_9999.pt" ]]; then
    echo "[ERROR] This script only exports model_9999.pt: $checkpoint_file" >&2
    return 1
  fi
  if [[ ! -f "$checkpoint_file" ]]; then
    echo "[ERROR] Checkpoint not found: $checkpoint_file" >&2
    return 1
  fi

  echo "[INFO] Exporting $task_id"
  echo "[INFO] Checkpoint: $checkpoint_file"

  uv run python - "$task_id" "$checkpoint_file" <<'PY'
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

  if checkpoint.name != "model_9999.pt":
    raise ValueError(f"expected model_9999.pt, got {checkpoint.name}")
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
}

export_all_exp4() {
  shopt -s nullglob
  local checkpoints=(
    logs/rsl_rl/smp_exp4_body_velocity_ablation/*group*/model_9999.pt
  )
  shopt -u nullglob

  if [[ "${#checkpoints[@]}" -eq 0 ]]; then
    echo "[ERROR] No exp4 model_9999.pt checkpoints found under logs/rsl_rl/smp_exp4_body_velocity_ablation/" >&2
    return 1
  fi

  local checkpoint_file run_dir group task_id
  for checkpoint_file in "${checkpoints[@]}"; do
    run_dir="$(basename "$(dirname "$checkpoint_file")")"
    if [[ "$run_dir" =~ group([1-7])_ ]]; then
      group="${BASH_REMATCH[1]}"
      task_id="Smp-BodyVelocity-Exp4-Group${group}-G1"
      export_one "$task_id" "$checkpoint_file"
    else
      echo "[WARN] Cannot infer task id from run dir, skipping: $run_dir" >&2
    fi
  done
}

case "${1:-}" in
  -h|--help)
    usage
    ;;
  --all-exp4|"")
    export_all_exp4
    ;;
  *)
    if [[ "$#" -ne 2 ]]; then
      usage >&2
      exit 2
    fi
    export_one "$1" "$2"
    ;;
esac
