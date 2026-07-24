#!/usr/bin/env bash
set -euo pipefail

# Current Exp6 policy launcher. Groups 7-12 use the stop prior; groups 13-14
# retain their command configuration and replace only the prior with forward_stop_all.

CUDA_VISIBLE_DEVICES=0 uv run scripts/train.py Smp-Forward-Exp6-Group7-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --env.events.init-smp-state.params.ckpt-path=datasets/pretrain_ckpt/pretrained_forward_stop.pt \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp6_forward_stop \
  --agent.run-name=group7_zero_prob_0p3

CUDA_VISIBLE_DEVICES=1 uv run scripts/train.py Smp-Forward-Exp6-Group8-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --env.events.init-smp-state.params.ckpt-path=datasets/pretrain_ckpt/pretrained_forward_stop.pt \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp6_forward_stop \
  --agent.run-name=group8_zero_prob_0p4

CUDA_VISIBLE_DEVICES=2 uv run scripts/train.py Smp-Forward-Exp6-Group9-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --env.events.init-smp-state.params.ckpt-path=datasets/pretrain_ckpt/pretrained_forward_stop.pt \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp6_forward_stop \
  --agent.run-name=group9_zero_prob_0p5

CUDA_VISIBLE_DEVICES=3 uv run scripts/train.py Smp-Forward-Exp6-Group10-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --env.events.init-smp-state.params.ckpt-path=datasets/pretrain_ckpt/pretrained_forward_stop.pt \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp6_forward_stop \
  --agent.run-name=group10_zero_0p3_low_0p35_high_0p35

CUDA_VISIBLE_DEVICES=4 uv run scripts/train.py Smp-Forward-Exp6-Group11-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --env.events.init-smp-state.params.ckpt-path=datasets/pretrain_ckpt/pretrained_forward_stop.pt \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp6_forward_stop \
  --agent.run-name=group11_zero_0p4_low_0p3_high_0p3

CUDA_VISIBLE_DEVICES=5 uv run scripts/train.py Smp-Forward-Exp6-Group12-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --env.events.init-smp-state.params.ckpt-path=datasets/pretrain_ckpt/pretrained_forward_stop.pt \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp6_forward_stop \
  --agent.run-name=group12_zero_0p5_low_0p25_high_0p25

CUDA_VISIBLE_DEVICES=1 uv run scripts/train.py Smp-Forward-Exp6-Group13-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --env.events.init-smp-state.params.ckpt-path=datasets/pretrain_ckpt/forward_stop_all.pt \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp6_forward_stop \
  --agent.run-name=group13_group8_cmd_forward_stop_all_prior

CUDA_VISIBLE_DEVICES=2 uv run scripts/train.py Smp-Forward-Exp6-Group14-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --env.events.init-smp-state.params.ckpt-path=datasets/pretrain_ckpt/forward_stop_all.pt \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp6_forward_stop \
  --agent.run-name=group14_group11_cmd_forward_stop_all_prior
