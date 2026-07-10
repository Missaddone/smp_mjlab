#!/usr/bin/env bash
set -euo pipefail

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
