#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_exp15_groups1_13.sh <group> <gpu> <source_wandb_run_path> \
    [checkpoint_name] [phase1_static_tilt_weight]

Groups 1-8 resume Exp13 G4. Groups 9-13 resume the Phase 1 group selected by
the user. The source checkpoint must be supplied explicitly for every group.
Diagnostics are written to logs/exp15_diagnostics/groupXX_diagnostics.csv.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 3 || $# -gt 5 ]]; then
  usage >&2
  exit 2
fi

GROUP="$1"
GPU="$2"
SOURCE_WANDB_PATH="$3"
CHECKPOINT_NAME="${4:-model_9999.pt}"
STATIC_TILT_WEIGHT="${5:-}"

case "$GROUP" in
  1) RUN_NAME="group01_control_exp13_g4_iter1000"; ITERATIONS=1000 ;;
  2) RUN_NAME="group02_control_exp13_g4_iter3000"; ITERATIONS=3000 ;;
  3) RUN_NAME="group03_finetune_exp13_g4_static_tilt_w005_iter1000"; ITERATIONS=1000 ;;
  4) RUN_NAME="group04_finetune_exp13_g4_static_tilt_w010_iter1000"; ITERATIONS=1000 ;;
  5) RUN_NAME="group05_finetune_exp13_g4_static_tilt_w020_iter1000"; ITERATIONS=1000 ;;
  6) RUN_NAME="group06_finetune_exp13_g4_static_tilt_w005_iter3000"; ITERATIONS=3000 ;;
  7) RUN_NAME="group07_finetune_exp13_g4_static_tilt_w010_iter3000"; ITERATIONS=3000 ;;
  8) RUN_NAME="group08_finetune_exp13_g4_static_tilt_w020_iter3000"; ITERATIONS=3000 ;;
  9) RUN_NAME="group09_phase2_control_selected_phase1_parent_iter1000"; ITERATIONS=1000 ;;
  10) RUN_NAME="group10_phase2_support_tilt_w002_iter1000"; ITERATIONS=1000 ;;
  11) RUN_NAME="group11_phase2_support_tilt_w005_iter1000"; ITERATIONS=1000 ;;
  12) RUN_NAME="group12_phase2_support_tilt_w002_iter3000"; ITERATIONS=3000 ;;
  13) RUN_NAME="group13_phase2_support_tilt_w005_iter3000"; ITERATIONS=3000 ;;
  *)
    echo "[ERROR] group must be an integer from 1 to 13. Got: $GROUP" >&2
    exit 2
    ;;
esac

STATIC_REWARD_ARGS=()
if (( GROUP >= 9 )); then
  if [[ -z "$STATIC_TILT_WEIGHT" ]]; then
    echo "[ERROR] Exp15 group $GROUP requires the selected Phase 1 static tilt weight." >&2
    usage >&2
    exit 2
  fi
  STATIC_REWARD_ARGS=(
    "--env.rewards.exp15_static_left_foot_tilt.weight=-${STATIC_TILT_WEIGHT}"
    "--env.rewards.exp15_static_right_foot_tilt.weight=-${STATIC_TILT_WEIGHT}"
  )
fi

WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}" CUDA_VISIBLE_DEVICES="$GPU" \
uv run scripts/train.py "Smp-BodyVelocity-Exp15-Group${GROUP}-G1" \
  --agent.max-iterations="$ITERATIONS" \
  --env.scene.num-envs=4096 \
  --agent.resume=True \
  --wandb-run-path="$SOURCE_WANDB_PATH" \
  --wandb-checkpoint-name="$CHECKPOINT_NAME" \
  "${STATIC_REWARD_ARGS[@]}" \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp15_body_velocity_flatfoot_staged \
  --agent.run-name="$RUN_NAME"
