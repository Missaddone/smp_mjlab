# Task Plan

## Goal
Design and then implement experiment 2 for the clean baseline:
`Smp-BodyVelocity-G1` with prior `datasets/pretrain_ckpt/amp_loco_clips2_mirrored_lafan_norm_12000.pt`.

Do not touch other task combinations unless the user explicitly changes this constraint.

## Current Constraints
- Base config: `4096` envs, `10000` iterations, single GPU.
- Command range: body-frame `x=-1..1`, `y=-1..1`, `yaw=-1..1`.
- Experiment 2 compares two reward variants:
  - Reward A: original task reward based on linear velocity error and yaw-rate error.
  - Reward B: original reward when command is non-static; when command requires static behavior, use a static-specific reward based on foot and root velocity error.
- User wants exact code-matching reward formulas before any code edits.
- After code changes are complete, provide train commands.

## Notion Flow Record Rules
- Use Notion database: `流程记录`, data source `collection://f2d5dc66-d42f-45cc-af2e-513bd60aa725`.
- `git的commit编号`: record the latest commit before the current experiment code changes, i.e. the previous run's commit.
- `代码改动`: concise summary of files changed and what changed.
- `需要达成的目的`: summarize the current experiment goal only.
- `效果`: user fills after running; if the user starts the next experiment and the previous effect is empty, remind them to fill it.
- `num-envs`: parallel env count.
- `task-name`: task name, normally e.g. `steering`, `forward`, or here `BodyVelocity`.
- `command设计`: command setup.
- `reward设计`: simplified formula with concrete parameter values.
- `速度区间`: command min/max velocity ranges and any special target-speed/static probability.
- `prior来源`: prior checkpoint path/source.

## Phases
- [complete] Phase 1: Confirm exact current reward and command formulas from code via subagent.
- [complete] Phase 2: Present Reward A and proposed Reward B formulas to user for confirmation.
- [complete] Phase 3: After confirmation, implement scoped code changes only for `Smp-BodyVelocity-G1` experiment variants.
- [complete] Phase 4: Verify CLI/help or tests and generate two training commands.
- [complete] Phase 5: Write Notion flow record with previous commit, parameters, code changes, and blank effect.

## Decisions
- 2026-06-30: User confirmed clean baseline is `Smp-BodyVelocity-G1` + `amp_loco_clips2_mirrored_lafan_norm_12000.pt`; ignore other task combinations.
- 2026-06-30: No code edits before user confirms exact Reward A and Reward B formulas.
- 2026-06-30: Experiment 2 uses two new tasks based only on the clean baseline:
  - `Smp-BodyVelocity-Exp2-G1`
  - `Smp-BodyVelocity-StaticExp2-G1`
- 2026-06-30: Experiment 2 static command probability defaults to `0.2`; static reward threshold defaults are `abs(cmd_x)<0.05`, `abs(cmd_y)<0.05`, `abs(cmd_yaw)<0.05 rad/s`.
