#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_exp11_body_velocity_foot_tilt.sh <group> <gpu> [wandb_run_path] [checkpoint_name]

Arguments:
  group             Exp11 group number, 1-26.
  gpu               CUDA device id for this foreground run.
  wandb_run_path    Optional source W&B run path. Defaults by group:
                    group1-4 and 9-17 -> robinbird-harbin-institute-of-technology/smp/xsxw4bwf
                    group5-8 and 18-26 -> robinbird-harbin-institute-of-technology/smp/4gvwh984
  checkpoint_name   Optional checkpoint name. Default: model_9999.pt.

Group mapping:
  1-4: base Exp10 group14, foot tilt weights -0.05, -0.10, -0.20, -0.30
  5-8: base Exp10 group15, foot tilt weights -0.05, -0.10, -0.20, -0.30
  9-17: base Exp10 group14, moving reward 0.6*r_l*r_y + 0.2*r_l + 0.2*r_y,
        fixed foot tilt -0.10, gait/drop penalty sweep
  18-26: base Exp10 group15, same group9-17 sweep
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
  9)
    TASK="Smp-BodyVelocity-Exp11-Group9-G1"
    RUN_NAME="group09_from_exp10_g14_moving_mix060_gait_w010"
    DEFAULT_WANDB_RUN_PATH="$EXP10_G14_WANDB_RUN_PATH"
    ;;
  10)
    TASK="Smp-BodyVelocity-Exp11-Group10-G1"
    RUN_NAME="group10_from_exp10_g14_moving_mix060_gait_w020"
    DEFAULT_WANDB_RUN_PATH="$EXP10_G14_WANDB_RUN_PATH"
    ;;
  11)
    TASK="Smp-BodyVelocity-Exp11-Group11-G1"
    RUN_NAME="group11_from_exp10_g14_moving_mix060_gait_w040"
    DEFAULT_WANDB_RUN_PATH="$EXP10_G14_WANDB_RUN_PATH"
    ;;
  12)
    TASK="Smp-BodyVelocity-Exp11-Group12-G1"
    RUN_NAME="group12_from_exp10_g14_moving_mix060_drop_w010"
    DEFAULT_WANDB_RUN_PATH="$EXP10_G14_WANDB_RUN_PATH"
    ;;
  13)
    TASK="Smp-BodyVelocity-Exp11-Group13-G1"
    RUN_NAME="group13_from_exp10_g14_moving_mix060_drop_w030"
    DEFAULT_WANDB_RUN_PATH="$EXP10_G14_WANDB_RUN_PATH"
    ;;
  14)
    TASK="Smp-BodyVelocity-Exp11-Group14-G1"
    RUN_NAME="group14_from_exp10_g14_moving_mix060_drop_w060"
    DEFAULT_WANDB_RUN_PATH="$EXP10_G14_WANDB_RUN_PATH"
    ;;
  15)
    TASK="Smp-BodyVelocity-Exp11-Group15-G1"
    RUN_NAME="group15_from_exp10_g14_moving_mix060_gait_w040_drop_w010"
    DEFAULT_WANDB_RUN_PATH="$EXP10_G14_WANDB_RUN_PATH"
    ;;
  16)
    TASK="Smp-BodyVelocity-Exp11-Group16-G1"
    RUN_NAME="group16_from_exp10_g14_moving_mix060_gait_w020_drop_w030"
    DEFAULT_WANDB_RUN_PATH="$EXP10_G14_WANDB_RUN_PATH"
    ;;
  17)
    TASK="Smp-BodyVelocity-Exp11-Group17-G1"
    RUN_NAME="group17_from_exp10_g14_moving_mix060_gait_w010_drop_w060"
    DEFAULT_WANDB_RUN_PATH="$EXP10_G14_WANDB_RUN_PATH"
    ;;
  18)
    TASK="Smp-BodyVelocity-Exp11-Group18-G1"
    RUN_NAME="group18_from_exp10_g15_moving_mix060_gait_w010"
    DEFAULT_WANDB_RUN_PATH="$EXP10_G15_WANDB_RUN_PATH"
    ;;
  19)
    TASK="Smp-BodyVelocity-Exp11-Group19-G1"
    RUN_NAME="group19_from_exp10_g15_moving_mix060_gait_w020"
    DEFAULT_WANDB_RUN_PATH="$EXP10_G15_WANDB_RUN_PATH"
    ;;
  20)
    TASK="Smp-BodyVelocity-Exp11-Group20-G1"
    RUN_NAME="group20_from_exp10_g15_moving_mix060_gait_w040"
    DEFAULT_WANDB_RUN_PATH="$EXP10_G15_WANDB_RUN_PATH"
    ;;
  21)
    TASK="Smp-BodyVelocity-Exp11-Group21-G1"
    RUN_NAME="group21_from_exp10_g15_moving_mix060_drop_w010"
    DEFAULT_WANDB_RUN_PATH="$EXP10_G15_WANDB_RUN_PATH"
    ;;
  22)
    TASK="Smp-BodyVelocity-Exp11-Group22-G1"
    RUN_NAME="group22_from_exp10_g15_moving_mix060_drop_w030"
    DEFAULT_WANDB_RUN_PATH="$EXP10_G15_WANDB_RUN_PATH"
    ;;
  23)
    TASK="Smp-BodyVelocity-Exp11-Group23-G1"
    RUN_NAME="group23_from_exp10_g15_moving_mix060_drop_w060"
    DEFAULT_WANDB_RUN_PATH="$EXP10_G15_WANDB_RUN_PATH"
    ;;
  24)
    TASK="Smp-BodyVelocity-Exp11-Group24-G1"
    RUN_NAME="group24_from_exp10_g15_moving_mix060_gait_w040_drop_w010"
    DEFAULT_WANDB_RUN_PATH="$EXP10_G15_WANDB_RUN_PATH"
    ;;
  25)
    TASK="Smp-BodyVelocity-Exp11-Group25-G1"
    RUN_NAME="group25_from_exp10_g15_moving_mix060_gait_w020_drop_w030"
    DEFAULT_WANDB_RUN_PATH="$EXP10_G15_WANDB_RUN_PATH"
    ;;
  26)
    TASK="Smp-BodyVelocity-Exp11-Group26-G1"
    RUN_NAME="group26_from_exp10_g15_moving_mix060_gait_w010_drop_w060"
    DEFAULT_WANDB_RUN_PATH="$EXP10_G15_WANDB_RUN_PATH"
    ;;
  *)
    echo "[ERROR] group must be 1-26, got: $GROUP" >&2
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
