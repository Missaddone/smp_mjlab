#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_exp13_body_velocity_moving_reward_mix.sh <group> <gpu>

Runs one Experiment 13 policy training job.

Group mapping:
  1: Exp10 group14, moving reward 0.5*r_l*r_y + 0.25*r_l + 0.25*r_y
  2: Exp10 group14, moving reward 0.6*r_l*r_y + 0.20*r_l + 0.20*r_y
  3: Exp10 group14, moving reward 0.7*r_l*r_y + 0.15*r_l + 0.15*r_y
  4: Exp10 group15, moving reward 0.5*r_l*r_y + 0.25*r_l + 0.25*r_y
  5: Exp10 group15, moving reward 0.6*r_l*r_y + 0.20*r_l + 0.20*r_y
  6: Exp10 group15, moving reward 0.7*r_l*r_y + 0.15*r_l + 0.15*r_y
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [ "$#" -ne 2 ]; then
  usage >&2
  exit 1
fi

GROUP="$1"
GPU="$2"

case "$GROUP" in
  1) RUN_NAME="group01_from_exp10_g14_moving_mix050_025_025" ;;
  2) RUN_NAME="group02_from_exp10_g14_moving_mix060_020_020" ;;
  3) RUN_NAME="group03_from_exp10_g14_moving_mix070_015_015" ;;
  4) RUN_NAME="group04_from_exp10_g15_moving_mix050_025_025" ;;
  5) RUN_NAME="group05_from_exp10_g15_moving_mix060_020_020" ;;
  6) RUN_NAME="group06_from_exp10_g15_moving_mix070_015_015" ;;
  *)
    echo "[ERROR] group must be an integer from 1 to 6. Got: $GROUP" >&2
    usage >&2
    exit 1
    ;;
esac

WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}" CUDA_VISIBLE_DEVICES="$GPU" \
uv run scripts/train.py "Smp-BodyVelocity-Exp13-Group${GROUP}-G1" \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp13_body_velocity_moving_reward_mix \
  --agent.run-name="$RUN_NAME"
