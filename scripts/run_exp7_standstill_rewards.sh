#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 uv run scripts/train.py Smp-Forward-Exp7-Group1-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --env.events.init-smp-state.params.ckpt-path=datasets/pretrain_ckpt/pretrained_forward_stop.pt \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp7_forward_stop_standstill_reward \
  --agent.run-name=group1_stand_still_exp_w2

CUDA_VISIBLE_DEVICES=1 uv run scripts/train.py Smp-Forward-Exp7-Group2-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --env.events.init-smp-state.params.ckpt-path=datasets/pretrain_ckpt/pretrained_forward_stop.pt \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp7_forward_stop_standstill_reward \
  --agent.run-name=group2_stand_still_exp_w5

CUDA_VISIBLE_DEVICES=2 uv run scripts/train.py Smp-Forward-Exp7-Group3-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --env.events.init-smp-state.params.ckpt-path=datasets/pretrain_ckpt/pretrained_forward_stop.pt \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp7_forward_stop_standstill_reward \
  --agent.run-name=group3_stand_still_exp_w7

CUDA_VISIBLE_DEVICES=3 uv run scripts/train.py Smp-Forward-Exp7-Group4-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --env.events.init-smp-state.params.ckpt-path=datasets/pretrain_ckpt/pretrained_forward_stop.pt \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp7_forward_stop_standstill_reward \
  --agent.run-name=group4_stand_still_vel_wneg002

CUDA_VISIBLE_DEVICES=4 uv run scripts/train.py Smp-Forward-Exp7-Group5-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --env.events.init-smp-state.params.ckpt-path=datasets/pretrain_ckpt/pretrained_forward_stop.pt \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp7_forward_stop_standstill_reward \
  --agent.run-name=group5_stand_still_vel_wneg005

CUDA_VISIBLE_DEVICES=5 uv run scripts/train.py Smp-Forward-Exp7-Group6-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --env.events.init-smp-state.params.ckpt-path=datasets/pretrain_ckpt/pretrained_forward_stop.pt \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp7_forward_stop_standstill_reward \
  --agent.run-name=group6_stand_still_vel_wneg010

CUDA_VISIBLE_DEVICES=6 uv run scripts/train.py Smp-Forward-Exp7-Group7-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --env.events.init-smp-state.params.ckpt-path=datasets/pretrain_ckpt/pretrained_forward_stop.pt \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp7_forward_stop_standstill_reward \
  --agent.run-name=group7_stand_still_feet_wneg02

CUDA_VISIBLE_DEVICES=7 uv run scripts/train.py Smp-Forward-Exp7-Group8-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --env.events.init-smp-state.params.ckpt-path=datasets/pretrain_ckpt/pretrained_forward_stop.pt \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp7_forward_stop_standstill_reward \
  --agent.run-name=group8_stand_still_feet_wneg05

CUDA_VISIBLE_DEVICES=8 uv run scripts/train.py Smp-Forward-Exp7-Group9-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --env.events.init-smp-state.params.ckpt-path=datasets/pretrain_ckpt/pretrained_forward_stop.pt \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp7_forward_stop_standstill_reward \
  --agent.run-name=group9_stand_still_feet_wneg10

CUDA_VISIBLE_DEVICES=9 uv run scripts/train.py Smp-Forward-Exp7-Group10-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --env.events.init-smp-state.params.ckpt-path=datasets/pretrain_ckpt/pretrained_forward_stop.pt \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp7_forward_stop_standstill_reward \
  --agent.run-name=group10_stand_still_double_support_w025

CUDA_VISIBLE_DEVICES=10 uv run scripts/train.py Smp-Forward-Exp7-Group11-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --env.events.init-smp-state.params.ckpt-path=datasets/pretrain_ckpt/pretrained_forward_stop.pt \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp7_forward_stop_standstill_reward \
  --agent.run-name=group11_stand_still_double_support_w05

CUDA_VISIBLE_DEVICES=11 uv run scripts/train.py Smp-Forward-Exp7-Group12-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --env.events.init-smp-state.params.ckpt-path=datasets/pretrain_ckpt/pretrained_forward_stop.pt \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp7_forward_stop_standstill_reward \
  --agent.run-name=group12_stand_still_double_support_w10

