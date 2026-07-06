#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 uv run scripts/train.py Smp-BodyVelocity-Exp4-Group1-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp4_body_velocity_ablation \
  --agent.run-name=group1_mix_p07_l015_y015_kxy1_kyaw1 &

CUDA_VISIBLE_DEVICES=1 uv run scripts/train.py Smp-BodyVelocity-Exp4-Group2-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp4_body_velocity_ablation \
  --agent.run-name=group2_mix_p07_l015_y015_kxy2_kyaw1 &

CUDA_VISIBLE_DEVICES=2 uv run scripts/train.py Smp-BodyVelocity-Exp4-Group3-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp4_body_velocity_ablation \
  --agent.run-name=group3_mix_p07_l015_y015_kxy1_kyaw05 &

CUDA_VISIBLE_DEVICES=3 uv run scripts/train.py Smp-BodyVelocity-Exp4-Group4-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp4_body_velocity_ablation \
  --agent.run-name=group4_sum_l05_y05_kxy1_kyaw05 &

CUDA_VISIBLE_DEVICES=4 uv run scripts/train.py Smp-BodyVelocity-Exp4-Group5-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp4_body_velocity_ablation \
  --agent.run-name=group5_sum_l05_y05_kxy1_kyaw05_zero_neg_proj &

CUDA_VISIBLE_DEVICES=5 uv run scripts/train.py Smp-BodyVelocity-Exp4-Group6-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp4_body_velocity_ablation \
  --agent.run-name=group6_mix_p06_l02_y02_kxy1_kyaw1 &

CUDA_VISIBLE_DEVICES=6 uv run scripts/train.py Smp-BodyVelocity-Exp4-Group7-G1 \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=smp_exp4_body_velocity_ablation \
  --agent.run-name=group7_mix_p08_l01_y01_kxy1_kyaw1 &

wait
