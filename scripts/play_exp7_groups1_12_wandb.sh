#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/play_exp7_groups1_12_wandb.sh [--gpu N] [--num-envs N] [--video-length N] \
    <group_number> <wandb_run_path>

Example:
  bash scripts/play_exp7_groups1_12_wandb.sh --gpu 0 3 entity/smp/group3_run

Valid group_number values: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
EOF
}

PLAY_GPU=0
NUM_ENVS=1
VIDEO_LENGTH=1500

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpu)
      PLAY_GPU="$2"
      shift 2
      ;;
    --num-envs)
      NUM_ENVS="$2"
      shift 2
      ;;
    --video-length)
      VIDEO_LENGTH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "[ERROR] Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

if [ "$#" -ne 2 ]; then
  echo "[ERROR] Expected <group_number> and <wandb_run_path>, got $# arguments." >&2
  usage >&2
  exit 1
fi

GROUP="$1"
WANDB_RUN_PATH="$2"

case "$GROUP" in
  1|2|3|4|5|6|7|8|9|10|11|12)
    ;;
  *)
    echo "[ERROR] group_number must be one of: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12. Got: $GROUP" >&2
    exit 1
    ;;
esac

play_group() {
  local group="$1"
  local wandb_run_path="$2"

  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES="$PLAY_GPU" \
    uv run scripts/play.py "Smp-Forward-Exp7-Group${group}-G1" \
      --wandb-run-path "$wandb_run_path" \
      --num-envs "$NUM_ENVS" \
      --video True \
      --video-length "$VIDEO_LENGTH" \
      --viewer viser

  echo "[DONE] exp7_group${group}"
}

play_group "$GROUP" "$WANDB_RUN_PATH"
