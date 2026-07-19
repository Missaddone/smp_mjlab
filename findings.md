# smp_mjlab Findings

## Body Velocity Migration
- `src/smp/rl/tasks/body_velocity` was originally a copied steering task.
- The copied package still registered `Smp-Steering-G1` and `Smp-Forward-G1`; this was replaced with a single `Smp-BodyVelocity-G1` registration.
- `BodyVelocityCommand` and `BodyVelocityCommandCfg` belong in `src/smp/rl/tasks/body_velocity/mdp/commands.py`.
- The my-dev command logic includes `_resample_command`, `_update_metrics`, and `_debug_vis_impl`; these are command implementation details and belong in the MDP command file, not env cfg.
- Actor observation should remove `base_lin_vel`; critic keeps `base_lin_vel`.
- Reward term key should remain `task_smp_product` for comparability.

## Experiment 4 Reward Formula
- `r_l = exp(-2.0 * ||v_xy_body - v_xy_cmd_body||^2)`.
- `r_y = exp(-1.0 * (yaw_rate_body - yaw_rate_cmd)^2)`.
- `r_task = 0.75 * r_l + 0.25 * r_y`.
- `r_final = r_task * smp_guidance_reward(t=(8,15,22), ws=6.0)`.

## Verification
- `MPLCONFIGDIR=/tmp/mplconfig_smp_tests ./.venv/bin/python -m unittest tests.test_body_velocity_task` passed.
- `./.venv/bin/ruff check src/smp/rl/rewards.py src/smp/rl/tasks/body_velocity src/smp/rl/tasks/__init__.py tests/test_body_velocity_task.py` passed.

## Experiment 4 Seven-Group Runner
- `k_xy` maps to `lin_vel_err_scale`.
- `k_yaw` maps to `yaw_rate_err_scale`.
- The user rejected Python runtime launchers and wants direct shell commands using `uv run scripts/train.py`.
- Current source reward interface can express linear sums by overriding `task_terms.*.1` weights and `task_terms.*.2.*-err-scale` values.
- Current source reward interface cannot express `r_l * r_y` product terms through plain `train.py` CLI overrides.
- Without source changes, only group 4 is exactly runnable from the seven-group table; group 5 also requires projection clipping code.
- Implemented seven registered env cfg files instead:
  - `Smp-BodyVelocity-Exp4-Group1-G1`
  - `Smp-BodyVelocity-Exp4-Group2-G1`
  - `Smp-BodyVelocity-Exp4-Group3-G1`
  - `Smp-BodyVelocity-Exp4-Group4-G1`
  - `Smp-BodyVelocity-Exp4-Group5-G1`
  - `Smp-BodyVelocity-Exp4-Group6-G1`
  - `Smp-BodyVelocity-Exp4-Group7-G1`
- Group 5 uses `body_velocity_linear_tracking_zero_negative_projection`.
- Direct bash runner: `scripts/run_exp4_7groups.sh`.

## ONNX Export
- `my-dev` does not contain a real `scripts/rsl_rl/export_onnx.py` file; `notebook.md` only has an old command memo for that path.
- Current export API is `MjlabOnPolicyRunner.export_policy_to_onnx(...)` in the installed `mjlab` package.
- The ONNX export script is `scripts/export_onnx_9999.sh`.
- The script only accepts `model_9999.pt` checkpoints and writes `model_9999.onnx` in the same directory.
- Use the same task id that produced the checkpoint, because the runner rebuilds the environment and actor observation shape from that task config.
- Dead-zone deployment export script: `scripts/export_onnx_with_deadzone.sh`.
- `scripts/export_onnx_with_deadzone.sh` is deployment-oriented and only supports body-velocity policies whose actor command is 3D `[x, y, yaw]`.
- `scripts/export_onnx_with_deadzone.sh` keeps the original actor observation layout and wraps the exported actor with tensor preprocessing only at ONNX export time.
- Default dead-zone ONNX output path is next to the checkpoint as `<checkpoint_stem>_deadzone.onnx`.
- The wrapper automatically finds the actor `command` slice from `env.observation_manager.active_terms["actor"]` and `group_obs_term_dim["actor"]`.
- The wrapper rejects non-3D command slices, so `steering` policies are not exported for real-robot dead-zone deployment by this script.
- Body-velocity dead-zone logic expects command dim 3: `[v_x, v_y, yaw_rate]`, and zeros the full command slice when `||command|| < dead_zone`.
- The wrapper applies dead-zone before the actor's `obs_normalizer`, matching training-time raw-observation semantics.

## LAFAN CSV Viser Clipping
- `my-dev` contains `scripts/clip_csv_viewer.py`; current `reconstruct` branch did not have this script when checked on 2026-07-06.
- Tool command pattern: `uv run scripts/clip_csv_viewer.py --input datasets/csv/lafan/<file>.csv --input-fps 30 --output-fps 50`.
- The viewer starts a `viser.ViserServer()` and prints a browser URL.
- GUI controls: `Play / Pause`, `Set Start`, `Set End`, frame slider, `Export CSV`.
- Default export directory is `datasets/csv_clips`.
- Default export filename is `<input_stem>_clip_<start_row>_<end_row>.csv`; duplicate names get `_01`, `_02`, etc.
- Export writes original CSV rows, not interpolated rows, so the result can be passed to `scripts/csv_to_npz.py`.
- CSV layout expected by related tools: `base_pos(3), base_quat_wxyz(4), joint_pos(29)`.

## Forward Static Stop Prior Data
- User added `datasets/csv/forward/stop.csv`, but it has only 3 rows and is skipped by `scripts/csv_to_npz.py` because default `window_size=10` and the 30->50fps interpolation gives only 4 frames.
- Added `scripts/make_static_csv.py` to repeat one source pose into a longer static CSV.
- Default output is `datasets/csv/forward/stop_static.csv`.
- Default settings: source row `-1`, `seconds=30`, `fps=30`, `min_rows=300`; actual generated file has 900 rows.
- Verification: converting `datasets/csv/forward` to `/tmp/smp_forward_npz_check` produced `stop_static.npz` with shape `(1490, 10, 59)`; original `stop.csv` was skipped.
- Verification of `stop_static.npz`: `root_lin_vel_max_abs=0.0`, `root_ang_vel_max_abs=3.72e-07`.

## Experiment 6 Forward Command Ablations
- Original `Smp-Forward-G1` command remains unchanged: `tar_speed_min=0.5`, `tar_speed_max=5.0`, no zero probability, no low-speed bin, no dead zone.
- Added generic steering target-speed sampling parameters in `src/smp/rl/tasks/steering/mdp/commands.py`: `zero_speed_prob`, `low_speed_prob`, `low_speed_min`, `low_speed_max`, and `dead_zone_speed`; all default to non-active behavior.
- Registered `Smp-Forward-Exp6-Group4-G1`: `P(v=0)=0.2`, otherwise `v~Uniform(0,5)`.
- Registered `Smp-Forward-Exp6-Group5-G1`: `P(v=0)=0.2`, `P(v~Uniform(0,1.5))=0.4`, `P(v~Uniform(1.5,5))=0.4`.
- Registered `Smp-Forward-Exp6-Group6-G1`: sample `v~Uniform(0,5)`, then apply dead zone `v=0 if v<1.0`, otherwise keep sampled speed.
- Registered `Smp-Forward-Exp6-Group7-G1`: `P(v=0)=0.3`, otherwise `v~Uniform(0,5)`.
- Registered `Smp-Forward-Exp6-Group8-G1`: `P(v=0)=0.4`, otherwise `v~Uniform(0,5)`.
- Registered `Smp-Forward-Exp6-Group9-G1`: `P(v=0)=0.5`, otherwise `v~Uniform(0,5)`.
- Registered `Smp-Forward-Exp6-Group10-G1`: `P(v=0)=0.3`, `P(v~Uniform(0,1.5))=0.35`, `P(v~Uniform(1.5,5))=0.35`.
- Registered `Smp-Forward-Exp6-Group11-G1`: `P(v=0)=0.4`, `P(v~Uniform(0,1.5))=0.3`, `P(v~Uniform(1.5,5))=0.3`.
- Registered `Smp-Forward-Exp6-Group12-G1`: `P(v=0)=0.5`, `P(v~Uniform(0,1.5))=0.25`, `P(v~Uniform(1.5,5))=0.25`.
- Registered `Smp-Forward-Exp6-Group13-G1`: same command config as group8, intended to use `forward_stop_all.pt`.
- Registered `Smp-Forward-Exp6-Group14-G1`: same command config as group11, intended to use `forward_stop_all.pt`.
- Reward, prior default, observations, terminations, target/facing direction behavior all inherit from original forward config unless CLI overrides change ckpt path.
- Direct bash runner for groups 7-12: `scripts/run_exp6_groups7_12.sh`; it overrides prior to `datasets/pretrain_ckpt/pretrained_forward_stop.pt`.
- `forward_stop_all` prior uses 8 forward actions: walk/jog/run and their mirrors, plus `stop_static.csv` and generated `stop_static_mirror.csv`.
- `scripts/run_exp6_prepare_forward_stop_all_prior.sh` stages those 8 CSVs, converts to NPZ under `datasets/npz/exp6/forward_stop_all`, pretrains with the standard 10000 epoch/2-layer/no-EMA/d_model=128 setup, and copies `forward_stop_all.pt` into `datasets/pretrain_ckpt/`.

## Experiment 7 Forward Standstill Rewards
- Goal: make forward policy stop from walk/run by adding standstill reward terms on top of Experiment 6 group7 command sampling.
- Shared command config for all groups: forward-only target direction, `tar_speed_min=0.0`, `tar_speed_max=5.0`, `zero_speed_prob=0.3`, no random target direction.
- Shared prior override in the training script: `datasets/pretrain_ckpt/pretrained_forward_stop.pt`.
- Shared base reward term: `1.0 * steering_target_velocity(command_name="steering", vel_err_scale=0.5)`, still wrapped by `task_smp_product`.
- Standstill mask: `||tar_speed * tar_dir_w|| <= 0.2`.
- `stand_still_exp`: `exp(-sum(abs(q - q_default))) * standstill_mask`, with `q_default` loaded from first row, columns 7:36 of `datasets/csv/forward/stop_static.csv`.
- `stand_still_vel`: positive penalty `sum(abs(qd)) * standstill_mask`; env-cfg uses negative weights.
- `stand_still_feet_motion_penalty`: positive penalty `sum(||v_foot||) * standstill_mask` over `left_ankle_roll_link` and `right_ankle_roll_link`; env-cfg uses negative weights.
- `stand_still_double_support`: reward `1` when both left and right foot collision geoms have vertical net contact force over `2.0` while standstill mask is active.
- Group order:
  - group1-3: `stand_still_exp`, weights `2.0`, `5.0`, `7.0`.
  - group4-6: `stand_still_vel`, weights `-0.02`, `-0.05`, `-0.10`.
  - group7-9: `stand_still_feet_motion_penalty`, weights `-0.2`, `-0.5`, `-1.0`.
  - group10-12: `stand_still_double_support`, weights `0.25`, `0.5`, `1.0`.
- Code locations:
  - reward functions: `src/smp/rl/tasks/steering/mdp/rewards.py`.
  - exp7 cfg table/builders: `src/smp/rl/tasks/steering/forward_exp7_env_cfg.py`.
  - task registration: `src/smp/rl/tasks/steering/__init__.py`.
  - training script: `scripts/run_exp7_standstill_rewards.sh`.

## Experiment 9 Prior/Reward Grid
- Goal: continue the Experiment 7 forward-stop scene while comparing prior data composition and revised stop reward formulas.
- Shared command config: Experiment 6 group7 forward-only command, `tar_speed_min=0.0`, `tar_speed_max=5.0`, `zero_speed_prob=0.3`.
- `datasets/csv/forward/stop_static.csv` has 36 standard CSV columns only: `root_pos(3), root_quat_xyzw(4), joint_pos(29)`. It has no explicit body or joint velocity columns. Adjacent-frame max difference across all columns was `0.0`, so CSV->NPZ should produce zero velocities for the static clip.
- Prior combinations:
  - P1 `exp9_prior1_forward_walk_to_stop.pt`: original forward six CSVs + `walk_to_stop.csv`/`w_2_s.csv`.
  - P2 `exp9_prior2_forward_w2s_s2w.pt`: P1 + `stop_to_walk.csv`/`s_2_w.csv`.
  - P3 `exp9_prior3_forward_w2s_s2w_stop_static.pt`: P2 + `stop_static.csv`.
  - P4 `exp9_prior4_forward_stop_static.pt`: original forward six CSVs + `stop_static.csv`.
- Prior preparation mirror rule: after staging each raw prior combination, `scripts/run_exp9_prepare_priors.sh` checks every non-`_mirror.csv` action and auto-generates the missing matching `_mirror.csv` via `scripts/mirror_motion_data.py`; existing forward mirror files are skipped.
- Reward versions:
  - A: Exp7-G4 baseline, `r_task = r_vel - 0.02 * m_still * sum(|qdot|)`.
  - B: `r_task = r_vel * ((1-m_still) + m_still*r_joint_vel)`, `r_joint_vel=exp(-0.02*sum(|qdot|))`.
  - C: moving uses `r_vel`; stopping uses `0.6*r_root_stop + 0.4*r_joint_vel`.
  - D: moving uses `r_vel`; stopping uses `r_root_stop*r_joint_vel`.
  - E: moving uses `r_vel`; stopping uses `0.6*r_root_stop*r_joint_vel + 0.2*r_root_stop + 0.2*r_joint_vel`.
  - `r_root_stop=exp(-2.0*||v_root_xy||^2)`, `m_still = 1[||v_cmd|| <= 0.2]`.
- Registered task ids: `Smp-Forward-Exp9-Group1-G1` through `Smp-Forward-Exp9-Group20-G1`.
- Group mapping: group1-5 = P1 with reward A-E; group6-10 = P2 with reward A-E; group11-15 = P3 with reward A-E; group16-20 = P4 with reward A-E.
- Code locations:
  - new task reward helpers: `src/smp/rl/tasks/steering/mdp/rewards.py`.
  - exp9 cfg table/builders: `src/smp/rl/tasks/steering/forward_exp9_env_cfg.py`.
  - task registration: `src/smp/rl/tasks/steering/__init__.py`.
  - plan doc: `experiment_9_plan.md`.
  - prior script: `scripts/run_exp9_prepare_priors.sh`.
  - policy script: `scripts/run_exp9_policy_groups1_20.sh`; run one group at a time as `bash scripts/run_exp9_policy_groups1_20.sh <group> <gpu>`.
  - play script: `scripts/play_exp9_groups1_20_wandb.sh`; run one group at a time as `bash scripts/play_exp9_groups1_20_wandb.sh [--gpu N] <group> <wandb_run_path>`.
  - `scripts/play.py` does not accept `--env.events.init-smp-state.params.ckpt-path=...`; Exp9 play must rely on `Smp-Forward-Exp9-Group{n}-G1` task cfg to select the matching prior path.
- Partial results in `exp9_summary.md` as of 2026-07-17:
  - Best prior is P4 `forward + stop_static`; adding `stop_to_walk` in P2 is harmful and often collapses motion.
  - Best current groups are 18/19/20, especially 19/20: normal run/walk plus reduced stop shaking, but with relay behavior and upper-body high-frequency jitter.
  - "Relay behavior" means starting from rest requires command speed around 2 before normal gait appears; after gait starts, reducing command into 1-2 can keep walking.
  - Next reward direction should not add command-zero velocity tracking. The static branch should target root angular/vertical stability, upper-body pose/velocity damping, and action/joint-acc smoothness.
- Exp9 group21-33 extension:
  - Group21: P4 + reward F, moving uses `r_vel`, stopping uses `r_root_stop * r_joint_vel * r_root_ang`.
  - Group22: P4 + reward G, moving uses `r_vel`, stopping uses `r_root_stop * r_joint_vel * r_upper_joint_vel`.
  - Group23: P4 + reward H, moving uses `r_vel`, stopping uses `r_root_stop * r_joint_vel * r_action_smooth`.
  - Group24-26 repeat group21-23 with command dead zone `dead_zone_speed=1.0`, meaning sampled speeds below `1.0 m/s` are set to zero by the existing steering command sampler.
  - Group27-32 repeat group21-26 with P5 `exp9_prior5_forward_stop_static_low_walk.pt`.
  - Group33 is P5 + reward D baseline, matching group19's stop product formulation, so the low-speed prior effect can be isolated.
  - P5 is staged from the forward base data plus `stop_static.csv` plus `g1_low_walk.csv`; the preparation script still auto-generates missing mirror CSVs for all staged non-mirror clips.
- Exp9 new reward constants:
  - `r_root_stop=exp(-2.0*||v_root_xy||^2)`.
  - `r_joint_vel=exp(-0.02*sum(|qdot|))`.
  - `r_root_ang=exp(-1.0*||omega_root_xy||^2)`.
  - `r_upper_joint_vel=exp(-0.05*sum(|qdot_upper|))` over waist, shoulder, elbow, and wrist joints.
  - `r_action_smooth=exp(-0.25*||a_t-a_{t-1}||^2)`.

## Experiment 5 Steering Prior
- Goal: train/evaluate a prior for multi-direction walking without high speed; keep original `Smp-Steering-G1` command/reward/observation exactly aligned with `master`.
- `src/smp/rl/tasks/steering/steering_env_cfg.py` matches `master` for `Smp-Steering-G1`.
- Original steering command config: `rand_tar_dir=True`, `rand_face_dir=True`, `tar_speed_min=0.5`, `tar_speed_max=2.0`, `resampling_time_range=(3.0, 8.0)`.
- Original steering reward key: `task_smp_product`; task terms are `0.5 * steering_target_velocity(vel_err_scale=1.0) + 0.5 * steering_face_direction`, multiplied by SMP guidance inside `task_smp_product`.
- Current branch has extra optional exp6 speed-sampling fields in `src/smp/rl/tasks/steering/mdp/commands.py`; defaults are inactive, so `Smp-Steering-G1` behavior is equivalent to `master`, but file contents are not identical.
- Batch mirror utility exists in `my-dev:scripts/mirror_motion_data.py`, not in current branch or `master`.
- Important risk: README says input CSV quaternion columns are `x y z w`; the `my-dev` mirror script docstring says CSV layout uses `w x y z`. Do not blindly bulk-trust mirrored CSVs without correcting/validating quaternion handling and visualizing one mirrored clip.
- Prior group definitions for the new csv_clips comparison:
  - `exp5_group1_all_csv_clips`: all `datasets/csv_clips/*.csv`.
  - `exp5_group2_no_stop`: all `datasets/csv_clips/*.csv` except names matching `*stop*.csv` (`stop_static`, `stop2`, etc.).
  - `exp5_group3_no_stop2`: all `datasets/csv_clips/*.csv` except names matching `*stop2*.csv`; keeps `stop_static.csv`.
  - `exp5_group4_dir_back_forward_walk`: `low_dir_back*.csv` from `datasets/csv_clips` plus `datasets/csv/forward/g1_walk.csv` by default.
- Experiment 5 prior script writes NPZ to `datasets/npz/exp5/<group_name>` and final prior checkpoints to `datasets/pretrain_ckpt/<group_name>.pt`.
- Experiment 5 prior script now stages each group's selected source files under `datasets/exp5_prior_sources/<group_name>/raw`, mirrors them into `datasets/exp5_prior_sources/<group_name>/mirrored` with `--include-original`, and converts the mirrored directory to NPZ. Every prior group therefore contains both original and left-right mirrored clips by default.
- Experiment 5 policy script uses original `Smp-Steering-G1` and W&B experiment name `smp_exp5_steering_prior_compare`.

## Experiment 10 Steering/Body-Velocity Static Switch
- Goal: transfer stop behavior from forward to original steering-style command and body-frame velocity command.
- Shared prior checkpoint path: `datasets/pretrain_ckpt/exp10_loco_stop_static.pt`.
- Prior data script: `scripts/run_exp10_prepare_prior.sh --gpu <gpu>`.
- Prior data source: all CSVs in `datasets/csv/loco` plus `datasets/csv/forward/stop_static.csv`; missing mirrors are generated for staged non-mirror CSVs.
- Steering groups:
  - Registered task ids: `Smp-Steering-Exp10-Group1-G1` through `Smp-Steering-Exp10-Group12-G1`.
  - Base config inherits current original `Smp-Steering-G1`: random target direction, random face direction, original observation shape, original SMP product wrapper.
  - Non-static `r_move = 0.5*r_vel + 0.5*r_face`.
  - `zero_command_when_zero_speed=True` makes actor command observation all-zero whenever sampled target speed is zero.
- Body-velocity groups:
  - Registered task ids: `Smp-BodyVelocity-Exp10-Group13-G1` through `Smp-BodyVelocity-Exp10-Group24-G1`.
  - Base config inherits `Smp-BodyVelocity-G1`: body-frame `[v_x, v_y, yaw_rate]`, actor observation without `base_lin_vel`, body-velocity SMP product wrapper.
  - Non-static `r_move = 0.75*r_l + 0.25*r_y`.
  - `zero_command_prob` and `dead_zone_speed` were added to `BodyVelocityCommandCfg`; zero/dead-zone events set `[v_x, v_y, yaw_rate]` to exactly zero.
- Command variants:
  - Steering C1: `P(v=0)=0.3`, otherwise `target_speed~U(0,2)`.
  - Steering C2: `P(v=0)=0.3`, otherwise `target_speed~U(0,4)`.
  - Steering C3/C4: C1/C2 plus `dead_zone_speed=0.5`; only these groups turn low nonzero speeds into all-zero actor commands.
  - Body C1: `P(command=0)=0.3`, otherwise `x,y~U(-2,2)`, `yaw~U(-1,1)`.
  - Body C2: `P(command=0)=0.3`, otherwise `x,y~U(-4,4)`, `yaw~U(-2,2)`.
  - Body C3/C4: C1/C2 plus full command-vector dead zone `||[x,y,yaw]|| < 0.5`, which zeros all three command components only in dead-zone groups.
- Reward variants:
  - C: moving uses task-specific original reward; stopping uses `0.6*r_root_stop + 0.4*r_joint_vel`.
  - D: moving uses task-specific original reward; stopping uses `r_root_stop*r_joint_vel`.
  - E: moving uses task-specific original reward; stopping uses `0.6*r_root_stop*r_joint_vel + 0.2*r_root_stop + 0.2*r_joint_vel`.
  - Steering stillness uses `||target_speed*target_dir|| <= 0.5`.
  - Body-velocity stillness uses `||[x,y,yaw]|| < 0.5`.
  - `r_root_stop=exp(-2.0*||v_root_xy||^2)`, `r_joint_vel=exp(-0.02*sum(|qdot|))`.
- Dead zone was set to `0.5 m/s` because it matches the original steering lower target speed and is less aggressive than the Exp9 `1.0 m/s` dead zone for the narrower `0-2/0-4` steering range. In dead-zone groups, command generation and reward stillness use the same `0.5` semantics, so low commands are transformed to zero and do not receive moving rewards.
- Group order is command-major then reward-minor:
  - Steering group1-3 = C1 with reward C/D/E; group4-6 = C2; group7-9 = C3; group10-12 = C4.
  - Body group13-15 = C1 with reward C/D/E; group16-18 = C2; group19-21 = C3; group22-24 = C4.
