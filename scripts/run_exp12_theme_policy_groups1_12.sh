#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_exp12_theme_policy_groups1_12.sh <group> <gpu>

Runs one Experiment 12 policy training job.
All groups inherit Exp10 body-velocity group14 config, replace the theme prior,
replace the moving reward with 0.6*r_l*r_y + 0.2*r_l + 0.2*r_y,
and add support-foot-tilt, persistent-single-support, and double-air penalties.

Group mapping:
  1-4:   male theme,     weights -0.05, -0.10, -0.20, -0.30
  5-8:   female theme,   weights -0.05, -0.10, -0.20, -0.30
  9-12:  children theme, weights -0.05, -0.10, -0.20, -0.30
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
  1) TASK="Smp-BodyVelocity-Exp12-Group1-G1"; RUN_NAME="group01_theme_male_exp10_g14_moving_mix060_gait_support_foot_tilt_w005" ;;
  2) TASK="Smp-BodyVelocity-Exp12-Group2-G1"; RUN_NAME="group02_theme_male_exp10_g14_moving_mix060_gait_support_foot_tilt_w010" ;;
  3) TASK="Smp-BodyVelocity-Exp12-Group3-G1"; RUN_NAME="group03_theme_male_exp10_g14_moving_mix060_gait_support_foot_tilt_w020" ;;
  4) TASK="Smp-BodyVelocity-Exp12-Group4-G1"; RUN_NAME="group04_theme_male_exp10_g14_moving_mix060_gait_support_foot_tilt_w030" ;;
  5) TASK="Smp-BodyVelocity-Exp12-Group5-G1"; RUN_NAME="group05_theme_female_exp10_g14_moving_mix060_gait_support_foot_tilt_w005" ;;
  6) TASK="Smp-BodyVelocity-Exp12-Group6-G1"; RUN_NAME="group06_theme_female_exp10_g14_moving_mix060_gait_support_foot_tilt_w010" ;;
  7) TASK="Smp-BodyVelocity-Exp12-Group7-G1"; RUN_NAME="group07_theme_female_exp10_g14_moving_mix060_gait_support_foot_tilt_w020" ;;
  8) TASK="Smp-BodyVelocity-Exp12-Group8-G1"; RUN_NAME="group08_theme_female_exp10_g14_moving_mix060_gait_support_foot_tilt_w030" ;;
  9) TASK="Smp-BodyVelocity-Exp12-Group9-G1"; RUN_NAME="group09_theme_children_exp10_g14_moving_mix060_gait_support_foot_tilt_w005" ;;
  10) TASK="Smp-BodyVelocity-Exp12-Group10-G1"; RUN_NAME="group10_theme_children_exp10_g14_moving_mix060_gait_support_foot_tilt_w010" ;;
  11) TASK="Smp-BodyVelocity-Exp12-Group11-G1"; RUN_NAME="group11_theme_children_exp10_g14_moving_mix060_gait_support_foot_tilt_w020" ;;
  12) TASK="Smp-BodyVelocity-Exp12-Group12-G1"; RUN_NAME="group12_theme_children_exp10_g14_moving_mix060_gait_support_foot_tilt_w030" ;;
  *)
    echo "[ERROR] group must be an integer from 1 to 12. Got: $GROUP" >&2
    usage >&2
    exit 1
    ;;
esac

WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}" CUDA_VISIBLE_DEVICES="$GPU" \
uv run scripts/train.py "$TASK" \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp12_body_velocity_theme_prior \
  --agent.run-name="$RUN_NAME"
