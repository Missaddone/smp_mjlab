# Task Plan

## Goal
Maintain durable context for `smp_mjlab` experiment design, command/play
consistency, and Notion records so future turns can continue without relying on
chat memory.

## Operating Rules
- Every user message must be appended to the Notion page
  `smp_mjlab Codex 用户消息记录` (`38f79c70-e2a7-81d2-b84d-f9e18cace163`).
- For project structure / code logic / relationship questions that do not
  require edits, use a subagent with model `gpt-5.4` and reasoning `medium`.
- Refresh these planning files with `planning-with-files` every 6 user messages
  after the last refresh, and also whenever the user explicitly asks.
- When a major experiment design or code change is complete, record the
  experiment parameters in Notion database `流程记录`.
- If the user starts a new major experiment and the previous experiment row has
  blank `效果`, remind the user to fill in the result/effect note.
- Do not modify unrelated task combinations. The user has repeatedly asked for
  scoped changes only.

## Notion Flow Record Rules
- Database: `流程记录`, data source
  `collection://f2d5dc66-d42f-45cc-af2e-513bd60aa725`.
- Columns to fill: `编号`, `git的commit编号`, `代码改动`, `效果`,
  `需要达成的目的`, `max-iteration`, `num-envs`, `task-name`,
  `command设计`, `reward设计`, `速度区间`, `prior来源`.
- `git的commit编号`: use the latest commit before the current experiment code
  changes, i.e. the previous run's commit.
- `代码改动`: concise summary of what files/logic changed.
- `效果`: leave blank for user-run experiments; remind later if still blank.

## Stable Experiment Context
- Experiment 1 common config:
  - Task: `Smp-BodyVelocity-G1`.
  - Prior: project original loco prior
    `datasets/pretrain_ckpt/pretrained_loco.pt`.
  - Reward: original `body_velocity_task_smp_product`, i.e. linear velocity and
    yaw-rate tracking error reward.
  - Base config: `4096` envs, `10000` iterations, single GPU, W&B logger.
- Experiment 1 groups:
  - Group 1 `range_1_1_1`: x `[-1,1]`, y `[-1,1]`, yaw `[-1,1]`.
  - Group 2 `range_4_4_2`: x `[-4,4]`, y `[-4,4]`, yaw `[-2,2]`.
  - Group 3 `range_xneg4_4_y0_yaw0`: x `[-4,4]`, y `[0,0]`, yaw `[0,0]`.
  - Group 4 `range_x0_4_y0_yaw0`: x `[0,4]`, y `[0,0]`, yaw `[0,0]`.
- Experiment 2 clean baseline:
  - Task family built from `Smp-BodyVelocity-G1`.
  - Clean prior checkpoint:
    `datasets/pretrain_ckpt/amp_loco_clips2_mirrored_lafan_norm_12000.pt`.
  - Added tasks: `Smp-BodyVelocity-Exp2-G1` and
    `Smp-BodyVelocity-StaticExp2-G1`.
  - Static command probability default: `0.2`.
  - Static command threshold defaults: `abs(cmd_x)<0.05`,
    `abs(cmd_y)<0.05`, `abs(cmd_yaw)<0.05 rad/s`.

## Current Open Decision
Experiment 3 is temporarily prioritized before experiment 2 training starts.
Ignore prior isolated-workspace/script-launch approaches. Provide direct
README-style `uv run scripts/train.py` commands only.

Experiment 3 compares only prior checkpoints:
- `Smp-Forward-G1` baseline: `pretrained_loco.pt`.
- `Smp-Forward-G1` custom:
  `amp_loco_clips2_mirrored_lafan_norm_12000.pt`.
- `Smp-Steering-G1` baseline: `pretrained_lafan_run.pt`.
- `Smp-Steering-G1` custom:
  `amp_loco_clips2_mirrored_lafan_norm_12000.pt`.
- Each run uses one GPU, default assignment `0 1 2 3`.
- The train command must pass `--agent.max-iterations 10000`; do not edit task
  code for this experiment.

## Reusable Train Template
Use Experiment 1 group 3 as the command-override pattern, changing only
run-name and command ranges:

```bash
cd /home/tyj/test/formal/smp_mjlab

MUJOCO_GL=egl \
./.venv/bin/python scripts/train.py Smp-BodyVelocity-G1 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 10000 \
  --agent.experiment-name smp_exp1_body_velocity_loco \
  --agent.run-name range_xneg4_4_y0_yaw0 \
  --agent.logger wandb \
  --agent.wandb-project smp \
  --gpu-ids 0 \
  --env.events.init-smp-state.params.ckpt-path datasets/pretrain_ckpt/pretrained_loco.pt \
  --env.commands.steering.lin-vel-x-min -4.0 \
  --env.commands.steering.lin-vel-x-max 4.0 \
  --env.commands.steering.lin-vel-y-min 0.0 \
  --env.commands.steering.lin-vel-y-max 0.0 \
  --env.commands.steering.yaw-rate-min 0.0 \
  --env.commands.steering.yaw-rate-max 0.0
```

## Reusable Play Template
Use EGL and pass `--video True`. Current built-in play does not restore training
command ranges; use this only when current registered env config is acceptable,
or replace with the future exact-run wrapper.

```bash
cd /home/tyj/test/formal/smp_mjlab
mkdir -p /tmp/mplconfig_smp_play

MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID=0 \
MPLCONFIGDIR=/tmp/mplconfig_smp_play \
./.venv/bin/python scripts/play.py Smp-BodyVelocity-G1 \
  --checkpoint-file logs/rsl_rl/smp_exp1_body_velocity_loco/2026-06-30_12-39-16_range_xneg4_4_y0_yaw0/model_9999.pt \
  --viewer viser \
  --num-envs 1 \
  --video True \
  --video-length 1500 \
  --video-width 1280 \
  --video-height 720
```
