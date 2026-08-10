#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/play_exp16_groups1_3_wandb.sh [--gpu N] [--num-envs N] \
    [--video-length N] [--viewer auto|native|viser] <group_number> <wandb_run_path>

Valid group_number values: 1..3
EOF
}

PLAY_GPU=0
NUM_ENVS=1
VIDEO_LENGTH=1500
VIEWER=viser
while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpu) PLAY_GPU="$2"; shift 2 ;;
    --num-envs) NUM_ENVS="$2"; shift 2 ;;
    --video-length) VIDEO_LENGTH="$2"; shift 2 ;;
    --viewer) VIEWER="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    -*) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 1 ;;
    *) break ;;
  esac
done
if [ "$#" -ne 2 ]; then
  echo "[ERROR] Expected <group_number> and <wandb_run_path>, got $# arguments." >&2
  usage >&2
  exit 1
fi

GROUP="$1"
WANDB_RUN_PATH="$2"
if ! [[ "$GROUP" =~ ^[1-3]$ ]]; then
  echo "[ERROR] group_number must be an integer from 1 to 3. Got: $GROUP" >&2
  exit 1
fi

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES="$PLAY_GPU" \
  uv run scripts/play.py "Smp-BodyVelocity-Exp16-Group${GROUP}-G1" \
    --wandb-run-path "$WANDB_RUN_PATH" \
    --num-envs "$NUM_ENVS" \
    --video True \
    --video-length "$VIDEO_LENGTH" \
    --viewer "$VIEWER"

echo "[DONE] exp16_group${GROUP}"
