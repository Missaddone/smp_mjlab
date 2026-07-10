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
