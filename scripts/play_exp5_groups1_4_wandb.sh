#!/usr/bin/env bash
set -euo pipefail

PLAY_GPU="${PLAY_GPU:-0}"
VIDEO_LENGTH="${VIDEO_LENGTH:-1500}"
NUM_ENVS="${NUM_ENVS:-1}"

EXP5_GROUP1_WANDB_RUN_PATH="${EXP5_GROUP1_WANDB_RUN_PATH:-}"
EXP5_GROUP2_WANDB_RUN_PATH="${EXP5_GROUP2_WANDB_RUN_PATH:-}"
EXP5_GROUP3_WANDB_RUN_PATH="${EXP5_GROUP3_WANDB_RUN_PATH:-}"
EXP5_GROUP4_WANDB_RUN_PATH="${EXP5_GROUP4_WANDB_RUN_PATH:-}"

require_wandb_path() {
  local name="$1"
  local value="$2"
  if [ -z "$value" ]; then
    echo "[ERROR] Set $name before running this script." >&2
    exit 1
  fi
}

play_exp5_group() {
  local group_name="$1"
  local wandb_run_path="$2"

  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES="$PLAY_GPU" \
    uv run scripts/play.py Smp-Steering-G1 \
      --wandb-run-path "$wandb_run_path" \
      --num-envs "$NUM_ENVS" \
      --video True \
      --video-length "$VIDEO_LENGTH" \
      --viewer viser

  echo "[DONE] $group_name"
}

require_wandb_path "EXP5_GROUP1_WANDB_RUN_PATH" "$EXP5_GROUP1_WANDB_RUN_PATH"
require_wandb_path "EXP5_GROUP2_WANDB_RUN_PATH" "$EXP5_GROUP2_WANDB_RUN_PATH"
require_wandb_path "EXP5_GROUP3_WANDB_RUN_PATH" "$EXP5_GROUP3_WANDB_RUN_PATH"
require_wandb_path "EXP5_GROUP4_WANDB_RUN_PATH" "$EXP5_GROUP4_WANDB_RUN_PATH"

play_exp5_group "exp5_group1_all_csv_clips" "$EXP5_GROUP1_WANDB_RUN_PATH"
play_exp5_group "exp5_group2_no_stop" "$EXP5_GROUP2_WANDB_RUN_PATH"
play_exp5_group "exp5_group3_no_stop2" "$EXP5_GROUP3_WANDB_RUN_PATH"
play_exp5_group "exp5_group4_dir_back_forward_walk" "$EXP5_GROUP4_WANDB_RUN_PATH"
