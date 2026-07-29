#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_exp14_groups1_40.sh <group> <gpu> <source_wandb_run_path> [checkpoint_name]

Groups 1-30 independently fine-tune Exp13 G5 from model_9999.pt. Groups
31-40 fine-tune the specified Exp14 G17/G20-G23 source from that parent's
final model_12998.pt or model_15998.pt checkpoint.

Groups:
  1-16  Existing Exp14 static-flat-foot and 3-second contact-duty ablations.
 17-24  Moving support-foot tilt: F > 1N and 40/60ms contact debounce,
        w_tilt = 0.1/0.2 (3000/6000 each).
 25-30  Moving max-force-foot tilt: F_max > 1N, w_tilt = 0.1/0.2/0.4
        (3000/6000 each).
 31-40  The matching G17/G20-G23 moving support-foot tilt plus static G1
        double-foot tilt, for 1000/3000 iterations.
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
REQUESTED_CHECKPOINT_NAME="${4:-}"

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
  17) RUN_NAME="group17_finetune_exp13_g5_support_tilt_debounce0.04_w0.1_iter3000"; ITERATIONS=3000 ;;
  18) RUN_NAME="group18_finetune_exp13_g5_support_tilt_debounce0.04_w0.1_iter6000"; ITERATIONS=6000 ;;
  19) RUN_NAME="group19_finetune_exp13_g5_support_tilt_debounce0.04_w0.2_iter3000"; ITERATIONS=3000 ;;
  20) RUN_NAME="group20_finetune_exp13_g5_support_tilt_debounce0.04_w0.2_iter6000"; ITERATIONS=6000 ;;
  21) RUN_NAME="group21_finetune_exp13_g5_support_tilt_debounce0.06_w0.1_iter3000"; ITERATIONS=3000 ;;
  22) RUN_NAME="group22_finetune_exp13_g5_support_tilt_debounce0.06_w0.1_iter6000"; ITERATIONS=6000 ;;
  23) RUN_NAME="group23_finetune_exp13_g5_support_tilt_debounce0.06_w0.2_iter3000"; ITERATIONS=3000 ;;
  24) RUN_NAME="group24_finetune_exp13_g5_support_tilt_debounce0.06_w0.2_iter6000"; ITERATIONS=6000 ;;
  25) RUN_NAME="group25_finetune_exp13_g5_max_force_tilt_w0.1_iter3000"; ITERATIONS=3000 ;;
  26) RUN_NAME="group26_finetune_exp13_g5_max_force_tilt_w0.1_iter6000"; ITERATIONS=6000 ;;
  27) RUN_NAME="group27_finetune_exp13_g5_max_force_tilt_w0.2_iter3000"; ITERATIONS=3000 ;;
  28) RUN_NAME="group28_finetune_exp13_g5_max_force_tilt_w0.2_iter6000"; ITERATIONS=6000 ;;
  29) RUN_NAME="group29_finetune_exp13_g5_max_force_tilt_w0.4_iter3000"; ITERATIONS=3000 ;;
  30) RUN_NAME="group30_finetune_exp13_g5_max_force_tilt_w0.4_iter6000"; ITERATIONS=6000 ;;
  31) RUN_NAME="group31_finetune_exp14_g17_support_tilt_debounce0.04_w0.1_static_tilt_w0.2_iter1000"; ITERATIONS=1000 ;;
  32) RUN_NAME="group32_finetune_exp14_g17_support_tilt_debounce0.04_w0.1_static_tilt_w0.2_iter3000"; ITERATIONS=3000 ;;
  33) RUN_NAME="group33_finetune_exp14_g20_support_tilt_debounce0.04_w0.2_static_tilt_w0.2_iter1000"; ITERATIONS=1000 ;;
  34) RUN_NAME="group34_finetune_exp14_g20_support_tilt_debounce0.04_w0.2_static_tilt_w0.2_iter3000"; ITERATIONS=3000 ;;
  35) RUN_NAME="group35_finetune_exp14_g21_support_tilt_debounce0.06_w0.1_static_tilt_w0.2_iter1000"; ITERATIONS=1000 ;;
  36) RUN_NAME="group36_finetune_exp14_g21_support_tilt_debounce0.06_w0.1_static_tilt_w0.2_iter3000"; ITERATIONS=3000 ;;
  37) RUN_NAME="group37_finetune_exp14_g22_support_tilt_debounce0.06_w0.1_static_tilt_w0.2_iter1000"; ITERATIONS=1000 ;;
  38) RUN_NAME="group38_finetune_exp14_g22_support_tilt_debounce0.06_w0.1_static_tilt_w0.2_iter3000"; ITERATIONS=3000 ;;
  39) RUN_NAME="group39_finetune_exp14_g23_support_tilt_debounce0.06_w0.2_static_tilt_w0.2_iter1000"; ITERATIONS=1000 ;;
  40) RUN_NAME="group40_finetune_exp14_g23_support_tilt_debounce0.06_w0.2_static_tilt_w0.2_iter3000"; ITERATIONS=3000 ;;
  *)
    echo "[ERROR] group must be an integer from 1 to 40. Got: $GROUP" >&2
    exit 2
    ;;
esac

case "$GROUP" in
  31|32|35|36|39|40) DEFAULT_CHECKPOINT_NAME="model_12998.pt" ;;
  33|34|37|38) DEFAULT_CHECKPOINT_NAME="model_15998.pt" ;;
  *) DEFAULT_CHECKPOINT_NAME="model_9999.pt" ;;
esac
CHECKPOINT_NAME="${REQUESTED_CHECKPOINT_NAME:-$DEFAULT_CHECKPOINT_NAME}"

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
