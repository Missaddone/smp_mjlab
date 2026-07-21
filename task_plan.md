# smp_mjlab Experiment Context Plan

## Active Objective
Maintain experiment context and implement experiment-specific SMP task changes without mixing unrelated branches or priors.

## Current Phase
- Status: in progress
- Branch: reconstruct
- Current experiment: Experiment 10 steering/body-velocity static switch
- Task under construction: make stop behavior work for original steering-style commands and body-frame velocity commands

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
- Dead-zone deployment ONNX export script now exists at `scripts/export_onnx_with_deadzone.sh`; it supports body-velocity `[x,y,yaw]` policies only, accepts any checkpoint file, and writes `<checkpoint_stem>_deadzone.onnx` by default.
- Do not change actor observation definitions just for real-robot dead-zone behavior; keep simulation training behavior in command terms, and use the ONNX wrapper export for deployment-side embedded dead-zone preprocessing.
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
- Experiment 9 now has registered task ids `Smp-Forward-Exp9-Group1-G1` through `Smp-Forward-Exp9-Group33-G1`.
- Experiment 9 group21-26 extend the current best P4 prior:
  - group21: P4 + stop product + root roll/pitch angular velocity damping.
  - group22: P4 + stop product + upper-body joint velocity damping.
  - group23: P4 + stop product + action smoothness.
  - group24-26 repeat group21-23 with command dead zone `dead_zone_speed=1.0`.
- Experiment 9 group27-32 repeat group21-26 with P5 prior `datasets/pretrain_ckpt/exp9_prior5_forward_stop_static_low_walk.pt`.
- Experiment 9 group33 is P5 + group19-style stop-product baseline for isolating the low-speed prior effect.
- Experiment 9 scripts:
  - `scripts/run_exp9_prepare_priors.sh prior5` builds only the low-speed prior after `datasets/csv/forward/g1_low_walk.csv` is added.
  - `scripts/run_exp9_policy_groups1_33.sh <group> <gpu>` runs one foreground W&B training job.
  - `scripts/play_exp9_groups1_33_wandb.sh [--gpu N] <group> <wandb_run_path>` plays one W&B checkpoint with EGL video.
- Experiment 10 uses one prior for all groups: `datasets/pretrain_ckpt/exp10_loco_stop_static.pt`.
- Experiment 10 prior data: `datasets/csv/loco/*.csv + datasets/csv/forward/stop_static.csv`, with missing mirrors auto-generated by `scripts/run_exp10_prepare_prior.sh`.
- Experiment 10 group1-12 are steering tasks registered as `Smp-Steering-Exp10-Group{1..12}-G1`.
- Experiment 10 group13-24 are body-velocity tasks registered as `Smp-BodyVelocity-Exp10-Group{13..24}-G1`.
- Experiment 10 command variants:
  - C1: zero probability 0.3, speed/range 0-2.
  - C2: zero probability 0.3, speed/range 0-4.
  - C3: C1 + dead zone 0.5.
  - C4: C2 + dead zone 0.5.
- Experiment 10 dead-zone commands must also receive static reward:
  - Steering static reward mask uses `||target_speed*target_dir|| <= 0.5`.
  - Body-velocity static reward mask uses `||[x,y,yaw]|| < 0.5`.
  - Only C3/C4 dead-zone groups transform low nonzero commands into zero actor commands; C1/C2 keep low nonzero commands unchanged.
- Experiment 10 reward variants follow Exp9 group18-20:
  - C: stop sum `0.6*r_root_stop + 0.4*r_joint_vel`.
  - D: stop product `r_root_stop*r_joint_vel`.
  - E: mixed product `0.6*r_root_stop*r_joint_vel + 0.2*r_root_stop + 0.2*r_joint_vel`.
- Experiment 10 scripts:
  - `scripts/run_exp10_prepare_prior.sh --gpu <gpu>`.
  - `scripts/run_exp10_policy_groups1_24.sh <group> <gpu>`.
  - `scripts/play_exp10_groups1_24_wandb.sh [--gpu N] <group> <wandb_run_path>`.
- Experiment 11 focuses only on Exp10 body-velocity group14/group15 foot tilt fine-tuning:
  - task ids: `Smp-BodyVelocity-Exp11-Group{1..8}-G1`.
  - group1-4 inherit Exp10 group14 command/reward and add support-foot-tilt weights `-0.05,-0.1,-0.2,-0.3`.
  - group5-8 inherit Exp10 group15 command/reward and add support-foot-tilt weights `-0.05,-0.1,-0.2,-0.3`.
  - actor and critic observations are unchanged; the contact sensor is used only by the reward term.
  - training script: `scripts/run_exp11_body_velocity_foot_tilt.sh <group> <gpu> <wandb_run_path> [checkpoint_name]`.
- Experiment 12 focuses on theme/style priors while preserving Exp10 group14 policy config and sweeping foot-tilt weights:
  - task ids: `Smp-BodyVelocity-Exp12-Group{1..12}-G1`.
  - group1-4 use `datasets/pretrain_ckpt/exp12_theme_male.pt`.
  - group5-8 use `datasets/pretrain_ckpt/exp12_theme_female.pt`.
  - group9-12 use `datasets/pretrain_ckpt/exp12_theme_children.pt`.
  - each theme sweeps `support_foot_tilt` weights `-0.05,-0.1,-0.2,-0.3`.
  - all groups inherit Exp10 group14 command/reward/observations; only prior ckpt path and foot-tilt weight change.
  - scripts: `scripts/analyze_theme_command_ranges.py`, `scripts/run_exp12_prepare_theme_priors.sh`, `scripts/run_exp12_theme_policy_groups1_12.sh`.
