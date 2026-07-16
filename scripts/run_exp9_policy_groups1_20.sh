#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_exp9_policy_groups1_20.sh <group> <gpu>

Examples:
  bash scripts/run_exp9_policy_groups1_20.sh 1 0
  bash scripts/run_exp9_policy_groups1_20.sh 16 3

Runs one Experiment 9 policy training job:
  group: 1..20
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
  1)
    TASK="Smp-Forward-Exp9-Group1-G1"
    CKPT="datasets/pretrain_ckpt/exp9_prior1_forward_walk_to_stop.pt"
    RUN_NAME="group01_p1_a_exp7g4_joint_vel_penalty"
    ;;
  2)
    TASK="Smp-Forward-Exp9-Group2-G1"
    CKPT="datasets/pretrain_ckpt/exp9_prior1_forward_walk_to_stop.pt"
    RUN_NAME="group02_p1_b_joint_vel_gate"
    ;;
  3)
    TASK="Smp-Forward-Exp9-Group3-G1"
    CKPT="datasets/pretrain_ckpt/exp9_prior1_forward_walk_to_stop.pt"
    RUN_NAME="group03_p1_c_stop_sum_switch"
    ;;
  4)
    TASK="Smp-Forward-Exp9-Group4-G1"
    CKPT="datasets/pretrain_ckpt/exp9_prior1_forward_walk_to_stop.pt"
    RUN_NAME="group04_p1_d_stop_product_switch"
    ;;
  5)
    TASK="Smp-Forward-Exp9-Group5-G1"
    CKPT="datasets/pretrain_ckpt/exp9_prior1_forward_walk_to_stop.pt"
    RUN_NAME="group05_p1_e_stop_product_mix_switch"
    ;;
  6)
    TASK="Smp-Forward-Exp9-Group6-G1"
    CKPT="datasets/pretrain_ckpt/exp9_prior2_forward_w2s_s2w.pt"
    RUN_NAME="group06_p2_a_exp7g4_joint_vel_penalty"
    ;;
  7)
    TASK="Smp-Forward-Exp9-Group7-G1"
    CKPT="datasets/pretrain_ckpt/exp9_prior2_forward_w2s_s2w.pt"
    RUN_NAME="group07_p2_b_joint_vel_gate"
    ;;
  8)
    TASK="Smp-Forward-Exp9-Group8-G1"
    CKPT="datasets/pretrain_ckpt/exp9_prior2_forward_w2s_s2w.pt"
    RUN_NAME="group08_p2_c_stop_sum_switch"
    ;;
  9)
    TASK="Smp-Forward-Exp9-Group9-G1"
    CKPT="datasets/pretrain_ckpt/exp9_prior2_forward_w2s_s2w.pt"
    RUN_NAME="group09_p2_d_stop_product_switch"
    ;;
  10)
    TASK="Smp-Forward-Exp9-Group10-G1"
    CKPT="datasets/pretrain_ckpt/exp9_prior2_forward_w2s_s2w.pt"
    RUN_NAME="group10_p2_e_stop_product_mix_switch"
    ;;
  11)
    TASK="Smp-Forward-Exp9-Group11-G1"
    CKPT="datasets/pretrain_ckpt/exp9_prior3_forward_w2s_s2w_stop_static.pt"
    RUN_NAME="group11_p3_a_exp7g4_joint_vel_penalty"
    ;;
  12)
    TASK="Smp-Forward-Exp9-Group12-G1"
    CKPT="datasets/pretrain_ckpt/exp9_prior3_forward_w2s_s2w_stop_static.pt"
    RUN_NAME="group12_p3_b_joint_vel_gate"
    ;;
  13)
    TASK="Smp-Forward-Exp9-Group13-G1"
    CKPT="datasets/pretrain_ckpt/exp9_prior3_forward_w2s_s2w_stop_static.pt"
    RUN_NAME="group13_p3_c_stop_sum_switch"
    ;;
  14)
    TASK="Smp-Forward-Exp9-Group14-G1"
    CKPT="datasets/pretrain_ckpt/exp9_prior3_forward_w2s_s2w_stop_static.pt"
    RUN_NAME="group14_p3_d_stop_product_switch"
    ;;
  15)
    TASK="Smp-Forward-Exp9-Group15-G1"
    CKPT="datasets/pretrain_ckpt/exp9_prior3_forward_w2s_s2w_stop_static.pt"
    RUN_NAME="group15_p3_e_stop_product_mix_switch"
    ;;
  16)
    TASK="Smp-Forward-Exp9-Group16-G1"
    CKPT="datasets/pretrain_ckpt/exp9_prior4_forward_stop_static.pt"
    RUN_NAME="group16_p4_a_exp7g4_joint_vel_penalty"
    ;;
  17)
    TASK="Smp-Forward-Exp9-Group17-G1"
    CKPT="datasets/pretrain_ckpt/exp9_prior4_forward_stop_static.pt"
    RUN_NAME="group17_p4_b_joint_vel_gate"
    ;;
  18)
    TASK="Smp-Forward-Exp9-Group18-G1"
    CKPT="datasets/pretrain_ckpt/exp9_prior4_forward_stop_static.pt"
    RUN_NAME="group18_p4_c_stop_sum_switch"
    ;;
  19)
    TASK="Smp-Forward-Exp9-Group19-G1"
    CKPT="datasets/pretrain_ckpt/exp9_prior4_forward_stop_static.pt"
    RUN_NAME="group19_p4_d_stop_product_switch"
    ;;
  20)
    TASK="Smp-Forward-Exp9-Group20-G1"
    CKPT="datasets/pretrain_ckpt/exp9_prior4_forward_stop_static.pt"
    RUN_NAME="group20_p4_e_stop_product_mix_switch"
    ;;
  *)
    echo "[ERROR] group must be an integer from 1 to 20. Got: $GROUP" >&2
    usage >&2
    exit 1
    ;;
esac

CUDA_VISIBLE_DEVICES="$GPU" uv run scripts/train.py "$TASK" \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --env.events.init-smp-state.params.ckpt-path="$CKPT" \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp9_forward_stop_prior_reward \
  --agent.run-name="$RUN_NAME"
