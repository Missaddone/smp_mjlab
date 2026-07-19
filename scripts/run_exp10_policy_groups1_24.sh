#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_exp10_policy_groups1_24.sh <group> <gpu>

Runs one Experiment 10 policy training job:
  group 1..12  -> Smp-Steering-Exp10-Group{group}-G1
  group 13..24 -> Smp-BodyVelocity-Exp10-Group{group}-G1

Example:
  bash scripts/run_exp10_policy_groups1_24.sh 1 0
  bash scripts/run_exp10_policy_groups1_24.sh 13 1
EOF
}

if [ "$#" -ne 2 ]; then
  usage >&2
  exit 1
fi

GROUP="$1"
GPU="$2"
CKPT="datasets/pretrain_ckpt/exp10_loco_stop_static.pt"

case "$GROUP" in
  1) TASK="Smp-Steering-Exp10-Group1-G1"; RUN_NAME="group01_steering_c1_c_zero03_speed0_2_stop_sum_switch" ;;
  2) TASK="Smp-Steering-Exp10-Group2-G1"; RUN_NAME="group02_steering_c1_d_zero03_speed0_2_stop_product_switch" ;;
  3) TASK="Smp-Steering-Exp10-Group3-G1"; RUN_NAME="group03_steering_c1_e_zero03_speed0_2_stop_product_mix_switch" ;;
  4) TASK="Smp-Steering-Exp10-Group4-G1"; RUN_NAME="group04_steering_c2_c_zero03_speed0_4_stop_sum_switch" ;;
  5) TASK="Smp-Steering-Exp10-Group5-G1"; RUN_NAME="group05_steering_c2_d_zero03_speed0_4_stop_product_switch" ;;
  6) TASK="Smp-Steering-Exp10-Group6-G1"; RUN_NAME="group06_steering_c2_e_zero03_speed0_4_stop_product_mix_switch" ;;
  7) TASK="Smp-Steering-Exp10-Group7-G1"; RUN_NAME="group07_steering_c3_c_zero03_speed0_2_dz05_stop_sum_switch" ;;
  8) TASK="Smp-Steering-Exp10-Group8-G1"; RUN_NAME="group08_steering_c3_d_zero03_speed0_2_dz05_stop_product_switch" ;;
  9) TASK="Smp-Steering-Exp10-Group9-G1"; RUN_NAME="group09_steering_c3_e_zero03_speed0_2_dz05_stop_product_mix_switch" ;;
  10) TASK="Smp-Steering-Exp10-Group10-G1"; RUN_NAME="group10_steering_c4_c_zero03_speed0_4_dz05_stop_sum_switch" ;;
  11) TASK="Smp-Steering-Exp10-Group11-G1"; RUN_NAME="group11_steering_c4_d_zero03_speed0_4_dz05_stop_product_switch" ;;
  12) TASK="Smp-Steering-Exp10-Group12-G1"; RUN_NAME="group12_steering_c4_e_zero03_speed0_4_dz05_stop_product_mix_switch" ;;
  13) TASK="Smp-BodyVelocity-Exp10-Group13-G1"; RUN_NAME="group13_body_velocity_c1_c_zero03_xy2_yaw1_stop_sum_switch" ;;
  14) TASK="Smp-BodyVelocity-Exp10-Group14-G1"; RUN_NAME="group14_body_velocity_c1_d_zero03_xy2_yaw1_stop_product_switch" ;;
  15) TASK="Smp-BodyVelocity-Exp10-Group15-G1"; RUN_NAME="group15_body_velocity_c1_e_zero03_xy2_yaw1_stop_product_mix_switch" ;;
  16) TASK="Smp-BodyVelocity-Exp10-Group16-G1"; RUN_NAME="group16_body_velocity_c2_c_zero03_xy4_yaw2_stop_sum_switch" ;;
  17) TASK="Smp-BodyVelocity-Exp10-Group17-G1"; RUN_NAME="group17_body_velocity_c2_d_zero03_xy4_yaw2_stop_product_switch" ;;
  18) TASK="Smp-BodyVelocity-Exp10-Group18-G1"; RUN_NAME="group18_body_velocity_c2_e_zero03_xy4_yaw2_stop_product_mix_switch" ;;
  19) TASK="Smp-BodyVelocity-Exp10-Group19-G1"; RUN_NAME="group19_body_velocity_c3_c_zero03_xy2_yaw1_dz05_stop_sum_switch" ;;
  20) TASK="Smp-BodyVelocity-Exp10-Group20-G1"; RUN_NAME="group20_body_velocity_c3_d_zero03_xy2_yaw1_dz05_stop_product_switch" ;;
  21) TASK="Smp-BodyVelocity-Exp10-Group21-G1"; RUN_NAME="group21_body_velocity_c3_e_zero03_xy2_yaw1_dz05_stop_product_mix_switch" ;;
  22) TASK="Smp-BodyVelocity-Exp10-Group22-G1"; RUN_NAME="group22_body_velocity_c4_c_zero03_xy4_yaw2_dz05_stop_sum_switch" ;;
  23) TASK="Smp-BodyVelocity-Exp10-Group23-G1"; RUN_NAME="group23_body_velocity_c4_d_zero03_xy4_yaw2_dz05_stop_product_switch" ;;
  24) TASK="Smp-BodyVelocity-Exp10-Group24-G1"; RUN_NAME="group24_body_velocity_c4_e_zero03_xy4_yaw2_dz05_stop_product_mix_switch" ;;
  *)
    echo "[ERROR] group must be an integer from 1 to 24. Got: $GROUP" >&2
    usage >&2
    exit 1
    ;;
esac

WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}" CUDA_VISIBLE_DEVICES="$GPU" \
  uv run scripts/train.py "$TASK" \
    --agent.max-iterations=10000 \
    --env.scene.num-envs=4096 \
    --env.events.init-smp-state.params.ckpt-path="$CKPT" \
    --agent.logger=wandb \
    --agent.wandb-project=smp \
    --agent.experiment-name=smp_exp10_static_steering_body_velocity \
    --agent.run-name="$RUN_NAME"
