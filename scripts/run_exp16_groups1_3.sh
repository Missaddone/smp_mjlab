#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_exp16_groups1_3.sh <group> <gpu>

Train Exp16 groups 1--3 from scratch for 10000 iterations.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 2 ]]; then
  usage >&2
  exit 2
fi

GROUP="$1"
GPU="$2"
case "$GROUP" in
  1) RUN_NAME="group01_motebu_stop_static_exp13_g4_reward" ;;
  2) RUN_NAME="group02_motebu_stop_static_exp13_g5_reward" ;;
  3) RUN_NAME="group03_motebu_stop_static_exp13_g6_reward" ;;
  *) echo "[ERROR] group must be 1, 2, or 3. Got: $GROUP" >&2; exit 2 ;;
esac

WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}" CUDA_VISIBLE_DEVICES="$GPU" \
uv run scripts/train.py "Smp-BodyVelocity-Exp16-Group${GROUP}-G1" \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp16_body_velocity_motebu_stop_static \
  --agent.run-name="$RUN_NAME"
