#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_exp13_groups1_27.sh <group> <gpu> [parent_wandb_run_path] [checkpoint_name]

Groups 1-6 train from scratch for 10000 iterations.
Groups 7-27 fine-tune for 3000 iterations and require their parent W&B path:
  7-10, 19, 22-23 <- Exp13 G4
  11-14, 20, 24-25 <- Exp13 G5
  15-18, 21, 26-27 <- Exp13 G6

Groups 19-27 use -0.05 foot tilt only above 80N per foot, plus a static
per-under-80N-foot penalty. G19-G21 use -0.1; G22-G27 sweep -0.05 and -0.2.
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
SOURCE_WANDB_PATH="${3:-}"
CHECKPOINT_NAME="${4:-model_9999.pt}"

case "$GROUP" in
  1) RUN_NAME="group01_from_exp10_g14_moving_mix050_025_025"; ITERATIONS=10000 ;;
  2) RUN_NAME="group02_from_exp10_g14_moving_mix060_020_020"; ITERATIONS=10000 ;;
  3) RUN_NAME="group03_from_exp10_g14_moving_mix070_015_015"; ITERATIONS=10000 ;;
  4) RUN_NAME="group04_from_exp10_g15_moving_mix050_025_025"; ITERATIONS=10000 ;;
  5) RUN_NAME="group05_from_exp10_g15_moving_mix060_020_020"; ITERATIONS=10000 ;;
  6) RUN_NAME="group06_from_exp10_g15_moving_mix070_015_015"; ITERATIONS=10000 ;;
  7) RUN_NAME="group07_finetune_exp13_g4_g20_style"; ITERATIONS=3000 ;;
  8) RUN_NAME="group08_finetune_exp13_g4_foot_tilt_w005"; ITERATIONS=3000 ;;
  9) RUN_NAME="group09_finetune_exp13_g4_foot_tilt_w020"; ITERATIONS=3000 ;;
  10) RUN_NAME="group10_finetune_exp13_g4_foot_tilt_w040"; ITERATIONS=3000 ;;
  11) RUN_NAME="group11_finetune_exp13_g5_g20_style"; ITERATIONS=3000 ;;
  12) RUN_NAME="group12_finetune_exp13_g5_foot_tilt_w005"; ITERATIONS=3000 ;;
  13) RUN_NAME="group13_finetune_exp13_g5_foot_tilt_w020"; ITERATIONS=3000 ;;
  14) RUN_NAME="group14_finetune_exp13_g5_foot_tilt_w040"; ITERATIONS=3000 ;;
  15) RUN_NAME="group15_finetune_exp13_g6_g20_style"; ITERATIONS=3000 ;;
  16) RUN_NAME="group16_finetune_exp13_g6_foot_tilt_w005"; ITERATIONS=3000 ;;
  17) RUN_NAME="group17_finetune_exp13_g6_foot_tilt_w020"; ITERATIONS=3000 ;;
  18) RUN_NAME="group18_finetune_exp13_g6_foot_tilt_w040"; ITERATIONS=3000 ;;
  19) RUN_NAME="group19_finetune_exp13_g4_foot_tilt80_static_double_support80"; ITERATIONS=3000 ;;
  20) RUN_NAME="group20_finetune_exp13_g5_foot_tilt80_static_double_support80"; ITERATIONS=3000 ;;
  21) RUN_NAME="group21_finetune_exp13_g6_foot_tilt80_static_double_support80"; ITERATIONS=3000 ;;
  22) RUN_NAME="group22_finetune_exp13_g4_foot_tilt80_static_double_support80_w005"; ITERATIONS=3000 ;;
  23) RUN_NAME="group23_finetune_exp13_g4_foot_tilt80_static_double_support80_w020"; ITERATIONS=3000 ;;
  24) RUN_NAME="group24_finetune_exp13_g5_foot_tilt80_static_double_support80_w005"; ITERATIONS=3000 ;;
  25) RUN_NAME="group25_finetune_exp13_g5_foot_tilt80_static_double_support80_w020"; ITERATIONS=3000 ;;
  26) RUN_NAME="group26_finetune_exp13_g6_foot_tilt80_static_double_support80_w005"; ITERATIONS=3000 ;;
  27) RUN_NAME="group27_finetune_exp13_g6_foot_tilt80_static_double_support80_w020"; ITERATIONS=3000 ;;
  *)
    echo "[ERROR] group must be an integer from 1 to 27. Got: $GROUP" >&2
    exit 2
    ;;
esac

EXTRA_ARGS=()
if (( GROUP >= 7 )); then
  if [[ -z "$SOURCE_WANDB_PATH" ]]; then
    echo "[ERROR] Exp13 group $GROUP requires a parent W&B run path." >&2
    usage >&2
    exit 2
  fi
  EXTRA_ARGS=(
    --agent.resume=True
    --wandb-run-path="$SOURCE_WANDB_PATH"
    --wandb-checkpoint-name="$CHECKPOINT_NAME"
  )
fi

WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}" CUDA_VISIBLE_DEVICES="$GPU" \
uv run scripts/train.py "Smp-BodyVelocity-Exp13-Group${GROUP}-G1" \
  --agent.max-iterations="$ITERATIONS" \
  --env.scene.num-envs=4096 \
  "${EXTRA_ARGS[@]}" \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp13_body_velocity_moving_reward_mix \
  --agent.run-name="$RUN_NAME"
