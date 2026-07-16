# smp_mjlab Experiment Context Plan

## Active Objective
Maintain experiment context and implement experiment-specific SMP task changes without mixing unrelated branches or priors.

## Current Phase
- Status: in progress
- Branch: reconstruct
- Current experiment: Experiment 9 forward stop prior/reward grid
- Task under construction: compare forward-stop prior compositions and stop reward formulas

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
- Experiment 6 now has forward command groups 4-14 registered under `Smp-Forward-Exp6-Group{4..14}-G1`.
- Experiment 6 group13/group14 are prior-isolation checks:
  - group13 reuses group8 command config and changes prior to `datasets/pretrain_ckpt/forward_stop_all.pt`.
  - group14 reuses group11 command config and changes prior to `datasets/pretrain_ckpt/forward_stop_all.pt`.
- Experiment 6 `forward_stop_all` prior script: `scripts/run_exp6_prepare_forward_stop_all_prior.sh`.
- Experiment 6 group13/14 training script: `scripts/run_exp6_groups13_14_forward_stop_all.sh`.
- Experiment 6 play script:
  - `scripts/play_exp6_groups7_12_wandb.sh` plays one requested group from positional arguments: `<group_number> <wandb_run_path>`; no `export` is required.
- Experiment 5 must use original master-style `Smp-Steering-G1` command/reward/observation; only prior should differ.
- Experiment 5 speed sampling is the original steering config: `tar_speed_min=0.5`, `tar_speed_max=2.0`, random target direction and random face direction.
- Experiment 5 scripts:
  - `scripts/run_exp5_prepare_priors.sh` prepares CSV subsets, mirrors each subset with originals included, converts the combined original+mirror CSVs to NPZ, pretrains priors, and copies final pt files into `datasets/pretrain_ckpt/`.
  - `scripts/run_exp5_policy_groups1_4.sh` trains original `Smp-Steering-G1` with only `ckpt-path` changed per group.
  - `scripts/play_exp5_groups1_4_wandb.sh` plays the four original `Smp-Steering-G1` exp5 policy runs from W&B run paths.
- Experiment 7 uses Experiment 6 group7 command as the base:
  - command: forward-only `target_speed`, `tar_speed_min=0.0`, `tar_speed_max=5.0`, `zero_speed_prob=0.3`.
  - prior override in training script: `datasets/pretrain_ckpt/pretrained_forward_stop.pt`.
  - base reward remains `task_smp_product` with `1.0 * steering_target_velocity(vel_err_scale=0.5)`.
  - standstill is active when `||target_speed * target_dir|| <= 0.2`.
  - default stand pose for `stand_still_exp` comes from first row joint columns of `datasets/csv/forward/stop_static.csv`.
  - registered task ids: `Smp-Forward-Exp7-Group1-G1` through `Smp-Forward-Exp7-Group12-G1`.
  - training script: `scripts/run_exp7_standstill_rewards.sh`.
