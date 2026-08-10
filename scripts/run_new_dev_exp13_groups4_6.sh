#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_new_dev_exp13_groups4_6.sh <group> <gpu>

Groups 4-6 train from scratch for 10000 iterations.  They are newly saved SMP
checkpoints: future NewDev Exp14 fine-tuning may resume them strictly with the
DiffNormalizer state stored in the checkpoint.
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
  4) RUN_NAME="new_dev_group04_from_exp10_g15_moving_mix050_025_025" ;;
  5) RUN_NAME="new_dev_group05_from_exp10_g15_moving_mix060_020_020" ;;
  6) RUN_NAME="new_dev_group06_from_exp10_g15_moving_mix070_015_015" ;;
  *)
    echo "[ERROR] group must be 4, 5, or 6. Got: $GROUP" >&2
    exit 2
    ;;
esac

SMP_CHECKPOINT_IDENTITY="new_dev_exp13_g${GROUP}" \
WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}" CUDA_VISIBLE_DEVICES="$GPU" \
uv run scripts/train.py "Smp-BodyVelocity-NewDev-Exp13-Group${GROUP}-G1" \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_new_dev_exp13_body_velocity_moving_reward_mix \
  --agent.run-name="$RUN_NAME"
