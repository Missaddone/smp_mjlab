#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/play_exp10_groups1_24_wandb.sh [--gpu N] [--num-envs N] [--video-length N] \
    <group_number> <wandb_run_path>

Examples:
  bash scripts/play_exp10_groups1_24_wandb.sh --gpu 0 1 entity/smp/group1_run
  bash scripts/play_exp10_groups1_24_wandb.sh --gpu 1 --video-length 1500 13 entity/smp/group13_run

Valid group_number values: 1..24
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
    TASK="Smp-Steering-Exp10-Group${GROUP}-G1"
    ;;
  13|14|15|16|17|18|19|20|21|22|23|24)
    TASK="Smp-BodyVelocity-Exp10-Group${GROUP}-G1"
    ;;
  *)
    echo "[ERROR] group_number must be an integer from 1 to 24. Got: $GROUP" >&2
    usage >&2
    exit 1
    ;;
esac

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES="$PLAY_GPU" \
  uv run scripts/play.py "$TASK" \
    --wandb-run-path "$WANDB_RUN_PATH" \
    --num-envs "$NUM_ENVS" \
    --video True \
    --video-length "$VIDEO_LENGTH" \
    --viewer viser

echo "[DONE] exp10_group${GROUP}"
