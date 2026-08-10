#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_new_dev_exp14_groups1_9.sh <group> <gpu> <source_wandb_run_path> [checkpoint]

Every group strictly resumes a new-dev Exp13 G4/G5/G6 checkpoint carrying its
SMP DiffNormalizer state.  The expected checkpoint identity mapping is:
  G1/G4/G7 <- Exp13 G4; G2/G5/G8 <- Exp13 G5; G3/G6/G9 <- Exp13 G6.
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
  1) EXPECTED_CHECKPOINT_IDENTITY="new_dev_exp13_g4"; RUN_NAME="group01_from_newdev_exp13_g4_control" ;;
  2) EXPECTED_CHECKPOINT_IDENTITY="new_dev_exp13_g5"; RUN_NAME="group02_from_newdev_exp13_g5_control" ;;
  3) EXPECTED_CHECKPOINT_IDENTITY="new_dev_exp13_g6"; RUN_NAME="group03_from_newdev_exp13_g6_control" ;;
  4) EXPECTED_CHECKPOINT_IDENTITY="new_dev_exp13_g4"; RUN_NAME="group04_from_newdev_exp13_g4_moving_debounced_support_tilt_w010" ;;
  5) EXPECTED_CHECKPOINT_IDENTITY="new_dev_exp13_g5"; RUN_NAME="group05_from_newdev_exp13_g5_moving_debounced_support_tilt_w010" ;;
  6) EXPECTED_CHECKPOINT_IDENTITY="new_dev_exp13_g6"; RUN_NAME="group06_from_newdev_exp13_g6_moving_debounced_support_tilt_w010" ;;
  7) EXPECTED_CHECKPOINT_IDENTITY="new_dev_exp13_g4"; RUN_NAME="group07_from_newdev_exp13_g4_static_double_foot_tilt_w010" ;;
  8) EXPECTED_CHECKPOINT_IDENTITY="new_dev_exp13_g5"; RUN_NAME="group08_from_newdev_exp13_g5_static_double_foot_tilt_w010" ;;
  9) EXPECTED_CHECKPOINT_IDENTITY="new_dev_exp13_g6"; RUN_NAME="group09_from_newdev_exp13_g6_static_double_foot_tilt_w010" ;;
  *)
    echo "[ERROR] group must be an integer from 1 to 9. Got: $GROUP" >&2
    exit 2
    ;;
esac

echo "[INFO] Expected checkpoint identity: $EXPECTED_CHECKPOINT_IDENTITY"

SMP_EXPECTED_CHECKPOINT_IDENTITY="$EXPECTED_CHECKPOINT_IDENTITY" \
WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}" CUDA_VISIBLE_DEVICES="$GPU" \
uv run scripts/train.py "Smp-BodyVelocity-NewDev-Exp14-Group${GROUP}-G1" \
  --agent.max-iterations=100 \
  --env.scene.num-envs=4096 \
  --agent.resume=True \
  --wandb-run-path="$SOURCE_WANDB_PATH" \
  --wandb-checkpoint-name="$CHECKPOINT_NAME" \
  --agent.algorithm.learning-rate=1e-4 \
  --agent.algorithm.desired-kl=0.002 \
  --agent.algorithm.num-learning-epochs=1 \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_new_dev_exp14_body_velocity_flatfoot \
  --agent.run-name="$RUN_NAME"
