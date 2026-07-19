#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_exp9_policy_groups1_33.sh <group> <gpu>

Examples:
  bash scripts/run_exp9_policy_groups1_33.sh 21 0
  bash scripts/run_exp9_policy_groups1_33.sh 30 3

Runs one Experiment 9 policy training job:
  group: 1..33
  gpu: CUDA_VISIBLE_DEVICES value for this one job
EOF
}

if [ "$#" -ne 2 ]; then
  usage >&2
  exit 1
fi

GROUP="$1"
GPU="$2"

case "$GROUP" in
  1|2|3|4|5)
    CKPT="datasets/pretrain_ckpt/exp9_prior1_forward_walk_to_stop.pt"
    ;;
  6|7|8|9|10)
    CKPT="datasets/pretrain_ckpt/exp9_prior2_forward_w2s_s2w.pt"
    ;;
  11|12|13|14|15)
    CKPT="datasets/pretrain_ckpt/exp9_prior3_forward_w2s_s2w_stop_static.pt"
    ;;
  16|17|18|19|20|21|22|23|24|25|26)
    CKPT="datasets/pretrain_ckpt/exp9_prior4_forward_stop_static.pt"
    ;;
  27|28|29|30|31|32|33)
    CKPT="datasets/pretrain_ckpt/exp9_prior5_forward_stop_static_low_walk.pt"
    ;;
  *)
    echo "[ERROR] group must be an integer from 1 to 33. Got: $GROUP" >&2
    usage >&2
    exit 1
    ;;
esac

case "$GROUP" in
  1) RUN_NAME="group01_p1_a_exp7g4_joint_vel_penalty" ;;
  2) RUN_NAME="group02_p1_b_joint_vel_gate" ;;
  3) RUN_NAME="group03_p1_c_stop_sum_switch" ;;
  4) RUN_NAME="group04_p1_d_stop_product_switch" ;;
  5) RUN_NAME="group05_p1_e_stop_product_mix_switch" ;;
  6) RUN_NAME="group06_p2_a_exp7g4_joint_vel_penalty" ;;
  7) RUN_NAME="group07_p2_b_joint_vel_gate" ;;
  8) RUN_NAME="group08_p2_c_stop_sum_switch" ;;
  9) RUN_NAME="group09_p2_d_stop_product_switch" ;;
  10) RUN_NAME="group10_p2_e_stop_product_mix_switch" ;;
  11) RUN_NAME="group11_p3_a_exp7g4_joint_vel_penalty" ;;
  12) RUN_NAME="group12_p3_b_joint_vel_gate" ;;
  13) RUN_NAME="group13_p3_c_stop_sum_switch" ;;
  14) RUN_NAME="group14_p3_d_stop_product_switch" ;;
  15) RUN_NAME="group15_p3_e_stop_product_mix_switch" ;;
  16) RUN_NAME="group16_p4_a_exp7g4_joint_vel_penalty" ;;
  17) RUN_NAME="group17_p4_b_joint_vel_gate" ;;
  18) RUN_NAME="group18_p4_c_stop_sum_switch" ;;
  19) RUN_NAME="group19_p4_d_stop_product_switch" ;;
  20) RUN_NAME="group20_p4_e_stop_product_mix_switch" ;;
  21) RUN_NAME="group21_p4_f_root_ang_stop" ;;
  22) RUN_NAME="group22_p4_g_upper_body_stop" ;;
  23) RUN_NAME="group23_p4_h_action_smooth_stop" ;;
  24) RUN_NAME="group24_p4_f_root_ang_stop_dz1" ;;
  25) RUN_NAME="group25_p4_g_upper_body_stop_dz1" ;;
  26) RUN_NAME="group26_p4_h_action_smooth_stop_dz1" ;;
  27) RUN_NAME="group27_p5_f_root_ang_stop" ;;
  28) RUN_NAME="group28_p5_g_upper_body_stop" ;;
  29) RUN_NAME="group29_p5_h_action_smooth_stop" ;;
  30) RUN_NAME="group30_p5_f_root_ang_stop_dz1" ;;
  31) RUN_NAME="group31_p5_g_upper_body_stop_dz1" ;;
  32) RUN_NAME="group32_p5_h_action_smooth_stop_dz1" ;;
  33) RUN_NAME="group33_p5_d_stop_product_switch_baseline" ;;
esac

CUDA_VISIBLE_DEVICES="$GPU" uv run scripts/train.py "Smp-Forward-Exp9-Group${GROUP}-G1" \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --env.events.init-smp-state.params.ckpt-path="$CKPT" \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp9_forward_stop_prior_reward \
  --agent.run-name="$RUN_NAME"
