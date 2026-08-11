# smp_mjlab Experiment Context Plan

## Active Objective
Maintain experiment context and implement experiment-specific SMP task changes without mixing unrelated branches or priors.

## Experiment 15: Tracking-Preserving Flat-Foot Investigation

### Goal
Explain and mitigate the tracking and gait-symmetry regression observed when
Exp14 flat-foot regularization is fine-tuned from the otherwise strong Exp13
G4 baseline. Determine whether high-frequency joint behavior is visible in
simulation and whether data mirroring or fine-tuning setup is a plausible
root cause.

### Phases
- [x] Phase 1: Restore Exp13/14 context and record observed Exp14 outcomes.
- [x] Phase 2: Audit Exp13 G4 parent config, Exp14 reward activation/magnitude,
  fine-tune checkpoint selection, and relevant motion-mirroring paths.
- [x] Phase 3: Propose and obtain approval for an Exp15 ablation plan. The plan
  must start from Exp13 G4 only and separate static flat-foot learning from
  moving flat-foot learning.
- [x] Phase 4: Implement approved task configs, metrics, train/play scripts,
  and documentation without modifying unrelated tasks.
- [x] Phase 5: Verify task builders/scripts and record exact limitations of
  simulator-side high-frequency-joint diagnostics.

## Current Phase
- Status: in progress
- Branch: reconstruct
- Current experiment: Experiment 11 evaluation and Experiment 13 follow-up design

## Documentation Maintenance: Experiment 15

### Goal
Create the Experiment 15 child document under the authorised Feishu master document, retaining the established experiment-page structure and recording only the implemented plan and pending training status.

### Phases
- [x] Phase 6: Verify the local Exp15 plan, implementation scope, and latest commits.
- [x] Phase 7: Read the Feishu parent tree and a comparable experiment page to preserve its page structure.
- [x] Phase 8: Create and populate the Experiment 15 child page, then read it back to verify the title, tables, and commands.

### Next Step
Await G1–G8 training and test results before selecting the Phase 2 parent group.

## Experiment 15 G1–G4 Degradation Analysis

### Goal
Use the recorded Exp15 training, replay, and diagnostic logs to determine why G1–G4 regress from the Exp13 G4 baseline; make no implementation change before establishing evidence.

### Phases
- [x] Phase 9: Locate and preserve the relevant G1–G4 and Exp13 G4 evidence.
- [x] Phase 10: Compare resume, optimization, command/reward, and diagnostic signals against the baseline.
- [x] Phase 11: State evidence-backed cause hypotheses and a minimal next experiment; do not modify task code or start more training.

### Next Step
Report the evidence-backed diagnosis to the user; keep Exp15 G5–G13 untrained.

## new_dev: Resumable SMP Fine-Tuning and Experiment 16

### Goal
Create an isolated `new_dev` branch without disturbing `reconstruct`; make SMP
DiffNormalizer state resumable in future PPO checkpoints, define a small Exp14
flat-foot fine-tuning suite using Exp13 G4--G6, and define Exp16 from only
`motebu`, `stop-static`, and their mirrored CSVs with body-velocity command
ranges x/y +/-1.5 and yaw +/-2.0.

### Phases
- [x] Phase 12: Audit the checkpoint/environment state boundary and existing test patterns; specify backward compatibility for old checkpoints.
- [x] Phase 13: Audit the new motion CSV, mirror/data-conversion/pretraining pipeline, and reusable Exp13/14 configuration patterns.
- [x] Phase 14: Present the isolated-branch design and compact experiment matrix for approval before implementation.
- [x] Phase 15: Create `new_dev`; add test-first resumable DiffNormalizer checkpoint support and scripts/configs for approved Exp14 and Exp16 work.
- [x] Phase 16: Run static/unit/script verification and hand off explicit training commands; do not start GPU training unless explicitly requested.

### Next Step
Run the explicitly supplied prior and policy commands on the training server;
record the resulting W&B paths before starting NewDev Exp14 fine-tunes.

## new_dev Exp16 Implementation Review

### Goal
Audit commit `1cd91ac` against the approved new_dev requirements, repair any
mandatory omissions, verify the complete checkpoint/prior/train/play workflow,
and update the established Feishu experiment documentation only after the code
review is clean.

### Phases
- [x] Phase 17: Review DiffNormalizer serialization and runner integration across train, resume, save, and play entrypoints.
- [x] Phase 18: Review fully mirrored NewDev Exp13 prior and Exp13 G4--G6 rerun pipeline.
- [x] Phase 19: Review Exp14 control/moving/static fine-tune matrix and strict parent-checkpoint handling.
- [x] Phase 20: Review Exp16 four-CSV prior, command/reward inheritance, launcher, train, and play scripts.
- [x] Phase 21: Implement mandatory fixes, run focused/full verification, and obtain a second code review.
- [x] Phase 22: Update Feishu using the established maintainer procedure and verify the written document.

### Next Step
None. The repaired implementation, local documentation, Notion logging, and
Feishu pages have all been verified; GPU training remains a user-run next step.
- Task under construction: none; Exp15 implementation and local cached-dependency verification are complete

## Exp15 Implementation Status

- Added `Smp-BodyVelocity-Exp15-Group1-G1` through `Group13-G1`.
- Added staged static/moving reward configs, metrics, CSV recorder, train/play scripts,
  launcher support, registry rows, and updated `exp15_summary.md`.
- CSV diagnostics are written to `logs/exp15_diagnostics/groupXX_diagnostics.csv`.
- Local syntax/static checks pass. Using the available cached dependency environment,
  Exp15 unit tests, recorder CSV smoke test, all 13 builders, and package import pass.

## Standing Rules
- Log every user message to Notion page `smp_mjlab Codex user message record`.
- For read-only project structure or code-logic questions, use a subagent with model `gpt-5.4` and reasoning `medium`.
- For explicitly delegated Feishu documentation work, use the local `FEISHU_DOCUMENT_MAINTAINER_SUBAGENT.md` procedure with standing write authorization and post-write re-fetch verification.
- Refresh these planning files every 6 user messages.
- Do not change prior, reward, command, task name, or code outside the requested scope.
- For experiment records, remind the user to fill prior experiment effects if a new experiment starts and Notion effect fields are blank.

## Command Templates
- Train commands use `CUDA_VISIBLE_DEVICES=<gpu> uv run scripts/train.py <task>`.
- Standard train flags: `--agent.max-iterations=10000`, `--env.scene.num-envs=4096`, `--agent.logger=wandb`, `--agent.wandb-project=smp`.
- Play commands use `uv run scripts/play.py ... --video True` and should include the same environment overrides as training when needed.
- For interactive body-velocity play, pass `--viewer viser`; the Viser command folder now exposes `lin_vel_x`, `lin_vel_y`, and `yaw_rate` sliders matching the active task's command ranges.
- W&B checkpoint replacement line: replace `--checkpoint-file ...` with `--wandb-run-path <org-or-entity>/smp/<wandb-run-id>`.
- Current `scripts/play.py` still does not accept CLI fixed body-velocity `[x,y,yaw]` commands, but Viser sliders are available for interactive body-velocity play.

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
- Documentation maintenance: create a short-term-problem section in Feishu `SMP-唐雨洁`; in parallel, extract skill/harness-related user messages from the Notion conversation log and have a product-manager subagent create the requested Notion resume-reference page.
- Exp13 tiptoe fine-tuning (implemented): fine-tune each Exp13 G4/G5/G6 `model_9999.pt` from its own W&B run, preserving that group's command, stop branch, prior, and moving-reward mix. Per parent, compare four variants: (a) retained Exp11 G20 regularization, `-0.1*support_foot_tilt -0.4*persistent_single_support`; (b) `-0.05*support_foot_tilt` only; (c) `-0.2*support_foot_tilt` only; (d) `-0.4*support_foot_tilt` only. This yields Exp13 G7-G18. The G20-style variant is deliberately unchanged despite its gait-timing semantics.
- Play-script lifecycle cleanup (complete): each experiment now has one current policy train script and one current play script. Exp12 obsolete `groups1_12` scripts were removed; Exp6 groups 7-14 were consolidated into single current train/play scripts. Prior-preparation scripts remain separate because they do not train or play policies.
- W&B run-id registry migration (complete): the tracked root-level `wandb_run_registry.csv` (`exp,group,run_id`) is now the playback source across servers. Launcher training writes its generated id; playback reads the CSV first and backfills a blank entry from W&B by experiment tag and group.
- Experiment launcher work:
  - Completed: one command lists supported group counts, launches a selected group on a selected GPU in a managed tmux window, and records the generated W&B path locally.
  - Completed: new W&B run ids are generated through `WANDB_RUN_ID`; Exp11's separate source-checkpoint W&B path is preserved for fine-tuning.
  - Completed: added Exp11 one-group playback wrapper so Exp11 evaluation does not require reconstructing a raw `scripts/play.py` command.
  - Exp13 G19-G21 are implemented: retain G4/G5/G6 respectively and add only `-0.05*foot_tilt` for feet above 80N terrain net force.
- Script lifecycle rule: when an experiment's group range is extended, replace the prior train/play script of the same role rather than retaining an outdated group-range script. Existing unrelated user changes are not removed in this automation task.
- ONNX export script now exists at `scripts/export_onnx.sh` and accepts any policy checkpoint filename.
- Dead-zone deployment ONNX export script now exists at `scripts/export_onnx_with_deadzone.sh`; it supports body-velocity `[x,y,yaw]` policies only, accepts any checkpoint file, and writes `<checkpoint_stem>_deadzone.onnx` by default.
- Do not change actor observation definitions just for real-robot dead-zone behavior; keep simulation training behavior in command terms, and use the ONNX wrapper export for deployment-side embedded dead-zone preprocessing.
- If user asks to run training, use W&B and keep command/prior consistent with the current env cfg.
- If user asks to customize body-velocity play commands, use Viser sliders for interactive tests. CLI fixed command flags such as `--body-cmd-x/y/yaw` are not implemented.
- Experiment 4 was recorded to Notion flow record as item `4`; user still needs to fill actual effect.
- Experiment 6 now has forward command groups 4-14 registered under `Smp-Forward-Exp6-Group{4..14}-G1`.
- Experiment 6 group13/group14 are prior-isolation checks:
  - group13 reuses group8 command config and changes prior to `datasets/pretrain_ckpt/forward_stop_all.pt`.
  - group14 reuses group11 command config and changes prior to `datasets/pretrain_ckpt/forward_stop_all.pt`.
- Experiment 6 `forward_stop_all` prior script: `scripts/run_exp6_prepare_forward_stop_all_prior.sh`.
- Experiment 6 policy training script: `scripts/run_exp6_groups7_14.sh`.
- Experiment 6 play script:
  - `scripts/play_exp6_groups7_14_wandb.sh` plays one requested group from positional arguments: `<group_number> <wandb_run_path>`; no `export` is required.
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
  - task ids: `Smp-BodyVelocity-Exp11-Group{1..26}-G1`.
  - group1-4 inherit Exp10 group14 command/reward and add support-foot-tilt weights `-0.05,-0.1,-0.2,-0.3`.
  - group5-8 inherit Exp10 group15 command/reward and add support-foot-tilt weights `-0.05,-0.1,-0.2,-0.3`.
  - group9-17 inherit Exp10 group14, use moving reward `0.6*r_l*r_y + 0.2*r_l + 0.2*r_y`, fixed support-foot-tilt `-0.1`, and sweep gait/drop penalties.
  - group18-26 repeat group9-17 on Exp10 group15.
  - gait-only weights: `-0.1,-0.2,-0.4`; drop-only weights: `-0.1,-0.3,-0.6`; combined groups use `(-0.4,-0.1)`, `(-0.2,-0.3)`, `(-0.1,-0.6)`.
  - actor and critic observations are unchanged; the contact sensor is used only by the reward term.
  - training script: `scripts/run_exp11_body_velocity_foot_tilt.sh <group> <gpu> <wandb_run_path> [checkpoint_name]`.
- Experiment 12 focuses on theme/style priors while preserving Exp10 group14 policy config and sweeping foot-tilt weights:
  - task ids: `Smp-BodyVelocity-Exp12-Group{1..18}-G1`.
  - group1-4 use `datasets/pretrain_ckpt/exp12_theme_male.pt`.
  - group5-8 use `datasets/pretrain_ckpt/exp12_theme_female.pt`.
  - group9-12 use `datasets/pretrain_ckpt/exp12_theme_children.pt`.
  - each theme sweeps `support_foot_tilt` weights `-0.05,-0.1,-0.2,-0.3`.
  - all groups inherit Exp10 group14 command/observations and theme prior replacement.
  - moving reward is now product-mix `0.6*r_l*r_y + 0.2*r_l + 0.2*r_y`; stop reward remains Exp10 group14 `r_root_stop*r_joint_vel`.
  - all groups add `persistent_single_support` weight `-0.2` and `double_air` weight `-0.3`; foot-tilt weight still sweeps by group.
  - group13-15 inherit Exp10 group14 exactly and only replace prior with male/female/children.
  - group16-18 inherit Exp10 group15 exactly and only replace prior with male/female/children.
  - For group13-18, SMP reward follows the theme prior because `init_smp_state.ckpt_path` is set to that theme ckpt and `smp_guidance_reward` uses the loaded `_smp_bundle`.
  - scripts: `scripts/analyze_theme_command_ranges.py`, `scripts/run_exp12_prepare_theme_priors.sh`, `scripts/run_exp12_theme_policy_groups1_18.sh`, `scripts/play_exp12_groups1_18_wandb.sh`.
- Experiment 13 tests body-velocity moving reward product mixes on Exp10 group14/group15, then fine-tunes G4/G5/G6 with foot regularization:
  - task ids: `Smp-BodyVelocity-Exp13-Group{1..21}-G1`.
  - group1-3 inherit Exp10 group14 and use moving weights `(0.5,0.25,0.25)`, `(0.6,0.2,0.2)`, `(0.7,0.15,0.15)`.
  - group4-6 inherit Exp10 group15 with the same moving-weight sweep.
  - only the moving branch changes; command, prior, stop reward, observations, and SMP wrapper stay from the selected Exp10 base group.
  - G7-G18 retain their parent task settings and add, per parent, G20-style `(-0.1 foot tilt, -0.4 persistent support)` or foot-tilt-only `-0.05/-0.2/-0.4`.
  - G19-G21 retain G4/G5/G6 respectively and add only `-0.05*foot_tilt` for each foot over 80N.
  - scripts: `scripts/run_exp13_groups1_21.sh`, `scripts/play_exp13_groups1_21_wandb.sh`.
- Experiment 14 is implemented with 30 independent fine-tuning tasks. It uses only Exp13 G5 `model_9999.pt`; no Exp13 G4/G6 parent is used. Each reward configuration has separate 3000- and 6000-iteration variants, both starting from G5.
  - Phase 1, static flat-foot ablations: static branch only. Per-foot raw tilt is `t_i = ||(R(q_i)e_z)_{xy}||^2 = sin^2(theta_i)`. Additive `R=R0-w(t_L+t_R)` uses `w={0.2,0.5,1.0}` (three groups). Multiplicative `R=R0*exp(-t_L)*exp(-t_R)` is one fixed-strength group; it has no independent additive weight.
  - Phase 2, moving duty-balance ablations: 3-second fixed contact-history window with force threshold `1N`, active only under non-static command. Compare exponential `R=R0*exp(-2*d)` (one configuration) and additive `R=R0-w_g*d` with `w_g={0.05,0.1,0.2}` (three configurations). This measures left/right support-duty balance, not actual cadence/frequency.
  - `1N` was deliberately selected over `80N`: a still-tiptoeing foot can be lightly loaded while the other foot carries most weight, and must remain represented in the duty statistic.
  - G17-G24: moving raw tilt applies only to per-foot support mask `F>1N AND current_contact_time>0.04/0.06s`; tilt weights are `-0.1/-0.2`.
  - G25-G30: moving raw tilt applies only to the foot with larger terrain net-force magnitude, provided `F_max>1N`; tilt weights are `-0.1/-0.2/-0.4`.
  - G17-G30 intentionally exclude all Exp14 static double-foot tilt terms. G31-G40 are second-stage fine-tunes from G17/G20-G23, retaining each moving support-foot term and adding G1's static `-0.2*(t_left+t_right)` penalty. They compare 1000 and 3000 iterations. Canonical scripts are `scripts/run_exp14_groups1_40.sh` and `scripts/play_exp14_groups1_40_wandb.sh`.
  - The combined tilt+duty reward phase is deliberately deferred until Phase 1 and 2 results select viable strengths.
