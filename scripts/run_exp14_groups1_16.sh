#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_exp14_groups1_16.sh <group> <gpu> <exp13_g5_wandb_run_path> [checkpoint_name]

All groups independently fine-tune Exp13 G5 from model_9999.pt by default.

Groups:
  1-6   static additive per-foot tilt: w=0.2, 0.5, 1.0 (3000/6000 each)
  7-8   static product: R0*exp(-t_left)*exp(-t_right) (3000/6000)
  9-10  moving 3s/1N contact-duty product: R0*exp(-2*duty_error) (3000/6000)
 11-16  moving 3s/1N contact-duty additive: w=0.05, 0.1, 0.2 (3000/6000 each)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 3 || $# -gt 4 ]]; then
  usage >&2
  exit 2
fi

GROUP="$1"
GPU="$2"
SOURCE_WANDB_PATH="$3"
CHECKPOINT_NAME="${4:-model_9999.pt}"

case "$GROUP" in
  1) RUN_NAME="group01_finetune_exp13_g5_static_tilt_add_w0.2_iter3000"; ITERATIONS=3000 ;;
  2) RUN_NAME="group02_finetune_exp13_g5_static_tilt_add_w0.2_iter6000"; ITERATIONS=6000 ;;
  3) RUN_NAME="group03_finetune_exp13_g5_static_tilt_add_w0.5_iter3000"; ITERATIONS=3000 ;;
  4) RUN_NAME="group04_finetune_exp13_g5_static_tilt_add_w0.5_iter6000"; ITERATIONS=6000 ;;
  5) RUN_NAME="group05_finetune_exp13_g5_static_tilt_add_w1_iter3000"; ITERATIONS=3000 ;;
  6) RUN_NAME="group06_finetune_exp13_g5_static_tilt_add_w1_iter6000"; ITERATIONS=6000 ;;
  7) RUN_NAME="group07_finetune_exp13_g5_static_tilt_product_iter3000"; ITERATIONS=3000 ;;
  8) RUN_NAME="group08_finetune_exp13_g5_static_tilt_product_iter6000"; ITERATIONS=6000 ;;
  9) RUN_NAME="group09_finetune_exp13_g5_gait_duty_exp_k2_iter3000"; ITERATIONS=3000 ;;
  10) RUN_NAME="group10_finetune_exp13_g5_gait_duty_exp_k2_iter6000"; ITERATIONS=6000 ;;
  11) RUN_NAME="group11_finetune_exp13_g5_gait_duty_add_w0.05_iter3000"; ITERATIONS=3000 ;;
  12) RUN_NAME="group12_finetune_exp13_g5_gait_duty_add_w0.05_iter6000"; ITERATIONS=6000 ;;
  13) RUN_NAME="group13_finetune_exp13_g5_gait_duty_add_w0.1_iter3000"; ITERATIONS=3000 ;;
  14) RUN_NAME="group14_finetune_exp13_g5_gait_duty_add_w0.1_iter6000"; ITERATIONS=6000 ;;
  15) RUN_NAME="group15_finetune_exp13_g5_gait_duty_add_w0.2_iter3000"; ITERATIONS=3000 ;;
  16) RUN_NAME="group16_finetune_exp13_g5_gait_duty_add_w0.2_iter6000"; ITERATIONS=6000 ;;
  *)
    echo "[ERROR] group must be an integer from 1 to 16. Got: $GROUP" >&2
    exit 2
    ;;
esac

WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}" CUDA_VISIBLE_DEVICES="$GPU" \
uv run scripts/train.py "Smp-BodyVelocity-Exp14-Group${GROUP}-G1" \
  --agent.max-iterations="$ITERATIONS" \
  --env.scene.num-envs=4096 \
  --agent.resume=True \
  --wandb-run-path="$SOURCE_WANDB_PATH" \
  --wandb-checkpoint-name="$CHECKPOINT_NAME" \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp14_body_velocity_flatfoot_duty \
  --agent.run-name="$RUN_NAME"
