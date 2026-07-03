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
Experiment 2 is back on the main path. The user may keep jobs running from the
current `master` checkout and run the updated `tyj-test` branch from a separate
directory. Use `git worktree` for this case so the current checkout is not
switched.

Experiment 2 run tasks:
- `Smp-BodyVelocity-Exp2-G1`: unified linear/yaw reward, including static
  commands.
- `Smp-BodyVelocity-StaticExp2-G1`: same moving reward; static commands use
  unified root linear/yaw scales plus the extra foot velocity product penalty.
- Both use prior `datasets/pretrain_ckpt/amp_loco_clips2_mirrored_lafan_norm_12000.pt`,
  command `x/y/yaw in [-1, 1]`, `static_prob=0.2`, `4096` envs, `10000`
  iterations, W&B logger, one GPU each.

## Reusable Train Template
Use the verified `CUDA_VISIBLE_DEVICES=N uv run scripts/train.py ...` pattern.
Experiment-specific task defaults should be left in the task config unless the
user explicitly asks for command/reward overrides:

```bash
cd /home/tyj/test/formal/smp_mjlab

CUDA_VISIBLE_DEVICES=0 uv run scripts/train.py <TASK> \
  --agent.max-iterations=10000 \
  --env.scene.num-envs=4096 \
  --agent.logger=wandb \
  --agent.wandb-project=smp \
  --agent.experiment-name=<EXPERIMENT_NAME> \
  --agent.run-name=<RUN_NAME>
```

## Reusable Play Template
Use EGL and pass `--video True`. Current built-in play does not restore training
command ranges; use this only when current registered env config is acceptable,
or replace with the future exact-run wrapper.
Do not use local filesystem discovery such as `find` to reshape the answer when
logs are absent locally. Provide the standard expected path and mention if the
file is not present in the current local checkout.

```bash
cd /home/tyj/test/formal/smp_mjlab_tyj-test
mkdir -p /tmp/mplconfig_smp_play

CUDA_VISIBLE_DEVICES=4 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID=0 \
MPLCONFIGDIR=/tmp/mplconfig_smp_play \
uv run scripts/play.py <TASK> \
  --checkpoint-file logs/rsl_rl/<EXPERIMENT_NAME>/<TIMESTAMP>_<RUN_NAME>/model_9999.pt \
  --viewer viser \
  --num-envs 1 \
  --video True \
  --video-length 1500 \
  --video-width 1280 \
  --video-height 720
```

W&B model-source replacement for play commands:

```bash
  --wandb-run-path <org-or-entity>/<wandb-project>/<wandb-run-id>
```

Use this in place of the `--checkpoint-file ...` line, following the README
style `uv run scripts/play.py <TASK> --wandb-run-path <org>/<project>/<run>`.

For experiment 2 specifically:
- `<EXPERIMENT_NAME>` = `smp_exp2_body_velocity_static_switch`
- Base run: task `Smp-BodyVelocity-Exp2-G1`, env cfg function
  `g1_body_velocity_exp2_smp_env_cfg`, run name `base_linear_yaw_static_cmd`.
- Static reward run: task `Smp-BodyVelocity-StaticExp2-G1`, env cfg function
  `g1_body_velocity_static_exp2_smp_env_cfg`, run name
  `static_cmd_root_foot_vel`.

## Experiment 3 Prior Compare
Run four direct train commands. Keep task defaults unchanged; only change prior
checkpoint between baseline/custom for the same task. Use `CUDA_VISIBLE_DEVICES`
instead of `--gpu-ids`.

- Forward baseline: `Smp-Forward-G1`, prior `pretrained_loco.pt`.
- Forward custom: `Smp-Forward-G1`, prior
  `amp_loco_clips2_mirrored_lafan_norm_12000.pt`.
- Steering baseline: `Smp-Steering-G1`, prior `pretrained_lafan_run.pt`.
- Steering custom: `Smp-Steering-G1`, prior
  `amp_loco_clips2_mirrored_lafan_norm_12000.pt`.
