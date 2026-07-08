#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 uv run scripts/train.py Smp-Steering-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --env.events.init-smp-state.params.ckpt-path=datasets/pretrain_ckpt/exp5_group1_all_csv_clips.pt \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp5_steering_prior_compare \
  --agent.run-name=exp5_group1_all_csv_clips

CUDA_VISIBLE_DEVICES=1 uv run scripts/train.py Smp-Steering-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --env.events.init-smp-state.params.ckpt-path=datasets/pretrain_ckpt/exp5_group2_no_stop.pt \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp5_steering_prior_compare \
  --agent.run-name=exp5_group2_no_stop

CUDA_VISIBLE_DEVICES=2 uv run scripts/train.py Smp-Steering-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --env.events.init-smp-state.params.ckpt-path=datasets/pretrain_ckpt/exp5_group3_no_stop2.pt \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp5_steering_prior_compare \
  --agent.run-name=exp5_group3_no_stop2

CUDA_VISIBLE_DEVICES=3 uv run scripts/train.py Smp-Steering-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --env.events.init-smp-state.params.ckpt-path=datasets/pretrain_ckpt/exp5_group4_dir_back_forward_walk.pt \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp5_steering_prior_compare \
  --agent.run-name=exp5_group4_dir_back_forward_walk
