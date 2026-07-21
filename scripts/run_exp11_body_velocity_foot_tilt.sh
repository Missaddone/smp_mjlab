#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_exp11_body_velocity_foot_tilt.sh <group> <gpu> [wandb_run_path] [checkpoint_name]

Arguments:
  group             Exp11 group number, 1-8.
  gpu               CUDA device id for this foreground run.
  wandb_run_path    Optional source W&B run path. Defaults by group:
                    group1-4 -> robinbird-harbin-institute-of-technology/smp/xsxw4bwf
                    group5-8 -> robinbird-harbin-institute-of-technology/smp/4gvwh984
  checkpoint_name   Optional checkpoint name. Default: model_9999.pt.

Group mapping:
  1-4: base Exp10 group14, foot tilt weights -0.05, -0.10, -0.20, -0.30
  5-8: base Exp10 group15, foot tilt weights -0.05, -0.10, -0.20, -0.30
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 || $# -gt 4 ]]; then
  usage >&2
  exit 2
fi

GROUP="$1"
GPU="$2"
WANDB_RUN_PATH="${3:-}"
CHECKPOINT_NAME="${4:-model_9999.pt}"

EXP10_G14_WANDB_RUN_PATH="robinbird-harbin-institute-of-technology/smp/xsxw4bwf"
EXP10_G15_WANDB_RUN_PATH="robinbird-harbin-institute-of-technology/smp/4gvwh984"

case "$GROUP" in
  1)
    TASK="Smp-BodyVelocity-Exp11-Group1-G1"
    RUN_NAME="group01_from_exp10_g14_support_foot_tilt_w005"
    DEFAULT_WANDB_RUN_PATH="$EXP10_G14_WANDB_RUN_PATH"
    ;;
  2)
    TASK="Smp-BodyVelocity-Exp11-Group2-G1"
    RUN_NAME="group02_from_exp10_g14_support_foot_tilt_w010"
    DEFAULT_WANDB_RUN_PATH="$EXP10_G14_WANDB_RUN_PATH"
    ;;
  3)
    TASK="Smp-BodyVelocity-Exp11-Group3-G1"
    RUN_NAME="group03_from_exp10_g14_support_foot_tilt_w020"
    DEFAULT_WANDB_RUN_PATH="$EXP10_G14_WANDB_RUN_PATH"
    ;;
  4)
    TASK="Smp-BodyVelocity-Exp11-Group4-G1"
    RUN_NAME="group04_from_exp10_g14_support_foot_tilt_w030"
    DEFAULT_WANDB_RUN_PATH="$EXP10_G14_WANDB_RUN_PATH"
    ;;
  5)
    TASK="Smp-BodyVelocity-Exp11-Group5-G1"
    RUN_NAME="group05_from_exp10_g15_support_foot_tilt_w005"
    DEFAULT_WANDB_RUN_PATH="$EXP10_G15_WANDB_RUN_PATH"
    ;;
  6)
    TASK="Smp-BodyVelocity-Exp11-Group6-G1"
    RUN_NAME="group06_from_exp10_g15_support_foot_tilt_w010"
    DEFAULT_WANDB_RUN_PATH="$EXP10_G15_WANDB_RUN_PATH"
    ;;
  7)
    TASK="Smp-BodyVelocity-Exp11-Group7-G1"
    RUN_NAME="group07_from_exp10_g15_support_foot_tilt_w020"
    DEFAULT_WANDB_RUN_PATH="$EXP10_G15_WANDB_RUN_PATH"
    ;;
  8)
    TASK="Smp-BodyVelocity-Exp11-Group8-G1"
    RUN_NAME="group08_from_exp10_g15_support_foot_tilt_w030"
    DEFAULT_WANDB_RUN_PATH="$EXP10_G15_WANDB_RUN_PATH"
    ;;
  *)
    echo "[ERROR] group must be 1-8, got: $GROUP" >&2
    exit 2
    ;;
esac

WANDB_RUN_PATH="${WANDB_RUN_PATH:-$DEFAULT_WANDB_RUN_PATH}"

WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}" CUDA_VISIBLE_DEVICES="$GPU" \
uv run scripts/train.py "$TASK" \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --agent.resume=True \
  --wandb-run-path="$WANDB_RUN_PATH" \
  --wandb-checkpoint-name="$CHECKPOINT_NAME" \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp11_body_velocity_foot_tilt \
  --agent.run-name="$RUN_NAME"
