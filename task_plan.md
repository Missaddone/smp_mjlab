# smp_mjlab Experiment Context Plan

## Active Objective
Maintain experiment context and implement experiment-specific SMP task changes without mixing unrelated branches or priors.

## Current Phase
- Status: in progress
- Branch: reconstruct
- Current experiment: Experiment 6 forward stop command ablations
- Task under construction: forward stop-prior command sampling variants groups 7-12

## Standing Rules
- Log every user message to Notion page `smp_mjlab Codex user message record`.
- For read-only project structure or code-logic questions, use a subagent with model `gpt-5.4` and reasoning `medium`.
- Refresh these planning files every 6 user messages.
- Do not change prior, reward, command, task name, or code outside the requested scope.
- For experiment records, remind the user to fill prior experiment effects if a new experiment starts and Notion effect fields are blank.

## Command Templates
- Train commands use `CUDA_VISIBLE_DEVICES=<gpu> uv run scripts/train.py <task>`.
- Standard train flags: `--agent.max-iterations=10000`, `--env.scene.num-envs=4096`, `--agent.logger=wandb`, `--agent.wandb-project=smp`.
- Play commands use `uv run scripts/play.py ... --video True` and should include the same environment overrides as training when needed.
- W&B checkpoint replacement line: replace `--checkpoint-file ...` with `--wandb-run-path <org-or-entity>/smp/<wandb-run-id>`.

## Experiment 4 Design
- Goal: test body-frame command and body-velocity reward style in an isolated task.
- Task id: `Smp-BodyVelocity-G1`.
- Command: body-frame `[v_x, v_y, yaw_rate]`.
- Command range: `x=[-1, 2]`, `y=[-1, 1]`, `yaw=[-1, 1]`.
- Prior: unchanged from the copied steering task unless user explicitly asks otherwise.
- Current prior path: `datasets/pretrain_ckpt/pretrained_lafan_run.pt`.
- Reward split:
  - r-task components live in `src/smp/rl/tasks/body_velocity/mdp/rewards.py`.
  - SMP final product wrapper lives in `src/smp/rl/rewards.py`.
  - Do not put `r_final` composition in task MDP reward files.

## Open Items
- ONNX export script now exists at `scripts/export_onnx_9999.sh`.
- If user asks to run training, use W&B and keep command/prior consistent with the current env cfg.
- Experiment 4 was recorded to Notion flow record as item `4`; user still needs to fill actual effect.
- Experiment 6 now has forward command groups 4-12 registered under `Smp-Forward-Exp6-Group{4..12}-G1`.
- Experiment 5 must use original master-style `Smp-Steering-G1` command/reward/observation; only prior should differ.
- Experiment 5 speed sampling is the original steering config: `tar_speed_min=0.5`, `tar_speed_max=2.0`, random target direction and random face direction.
