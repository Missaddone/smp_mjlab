# smp_mjlab Findings

## Experiment 15 Intake (2026-08-05)
- User reports that Exp13 G4-G6 originally had strong tracking and symmetric
  motion, while Exp14 flat-foot fine-tunes often degraded tracking and gait.
- Reported Exp14 observations: forward G36 is acceptable; G31-G35 and G39
  have an impaired swing foot. Backward G31/G35/G36/G39 exhibit one-foot-ahead
  gait; G33/G34 add poor tracking. Lateral G31/G32/G35/G36/G39 show an
  asymmetric left-back/right-forward gait and may drift backward; G33/G34 also
  lose lateral tracking and cannot move right. G37/G38 are rejected because
  they still tiptoe while walking.
- Exp15 must start from Exp13 G4 only. The first requested hypothesis is a
  staged procedure: solve flat feet during standing first, then address flat
  feet while walking. The user also asks for independent hypotheses about
  high-frequency joint noise, fine-tuning validity, and whether source motion
  data were fully mirrored.

## Experiment 15 Audit Findings (in progress)
- Important correction: the checked-in Exp14 code does **not** inherit Exp13
  G4-G6. `body_velocity_exp14_env_cfg.py` hard-codes `_PARENT_GROUP = 5`, and
  `run_exp14_groups1_40.sh` states groups 1-30 fine-tune Exp13 G5. Thus the
  reported Exp14 outcomes are not a direct G4/G6 comparison unless a different
  uncommitted/remote runner was used.
- Exp13 G4 is `Exp10 G15 + moving reward 0.5*r_l*r_y + 0.25*r_l + 0.25*r_y`.
  Exp13 G5 instead uses `0.6*r_l*r_y + 0.2*r_l + 0.2*r_y`; this reward change
  alone makes them different policy starting points.
- Exp14 moving tilt groups use an additive reward, active only if command norm
  exceeds 0.2, terrain net-force is above 1N, and (for G17-G24) continuous
  contact exceeds 40/60 ms. Static terms use the same 0.2 command threshold.
  The second-stage G31-G40 therefore combine static and moving penalties in
  one fine-tune rather than first consolidating a static-only policy.
- The current mirror utility explicitly declares CSV quaternion input order
  `base_quat_xyzw`, mirrors quaternion x/z signs, swaps left/right joints and
  end-effectors, and applies roll/yaw sign changes. This is internally
  consistent with xyzw. It cannot prove that the historical prior used by
  Exp13 G4 was generated through this script or that the script's rigid joint
  layout matches every original source CSV; trace the actual prior preparation
  command and artifacts next.
- Exp13 G4/G5/G6 all use the same Exp10 prior
  `datasets/pretrain_ckpt/exp10_loco_stop_static.pt`; only their moving reward
  mix differs. The committed `run_exp10_prepare_prior.sh` is the prior builder
  relevant to this checkpoint and invokes `mirror_motion_data.py` with
  `--include-original` before CSV-to-NPZ and pretraining.
- No existing body-velocity reward term measures action-rate, action
  acceleration, torque, or motor power. The stop branch only applies a joint
  velocity component when the command is static. Therefore it cannot diagnose
  or suppress high-frequency joint oscillation while the robot is moving.

## Body Velocity Migration
- `src/smp/rl/tasks/body_velocity` was originally a copied steering task.
- The copied package still registered `Smp-Steering-G1` and `Smp-Forward-G1`; this was replaced with a single `Smp-BodyVelocity-G1` registration.
- `BodyVelocityCommand` and `BodyVelocityCommandCfg` belong in `src/smp/rl/tasks/body_velocity/mdp/commands.py`.
- The my-dev command logic includes `_resample_command`, `_update_metrics`, and `_debug_vis_impl`; these are command implementation details and belong in the MDP command file, not env cfg.
- Actor observation should remove `base_lin_vel`; critic keeps `base_lin_vel`.
- Reward term key should remain `task_smp_product` for comparability.
- Current body-velocity play cannot set fixed `[x,y,yaw]` from CLI. `scripts/play.py` only delegates to `mjlab.scripts.play.main`, and `play --help` exposes no command override flags.
- `BodyVelocityCommand._resample_command()` randomly samples `lin_vel_b[:,0]`, `lin_vel_b[:,1]`, and `yaw_rate` from cfg ranges.
- `BodyVelocityCommand` now implements Viser GUI sliders like steering: enable the command folder in Viser and adjust `lin_vel_x`, `lin_vel_y`, and `yaw_rate`. Slider min/max values are taken from the active task's command cfg.
- Minimal future CLI support would add optional fixed command fields to `BodyVelocityCommandCfg` and expose them through a project-local play wrapper.

## Experiment 4 Reward Formula
- `r_l = exp(-2.0 * ||v_xy_body - v_xy_cmd_body||^2)`.
- `r_y = exp(-1.0 * (yaw_rate_body - yaw_rate_cmd)^2)`.
- `r_task = 0.75 * r_l + 0.25 * r_y`.
- `r_final = r_task * smp_guidance_reward(t=(8,15,22), ws=6.0)`.

## Verification
- `MPLCONFIGDIR=/tmp/mplconfig_smp_tests ./.venv/bin/python -m unittest tests.test_body_velocity_task` passed.
- `./.venv/bin/ruff check src/smp/rl/rewards.py src/smp/rl/tasks/body_velocity src/smp/rl/tasks/__init__.py tests/test_body_velocity_task.py` passed.

## Shared W&B Run Registry
- `wandb_run_registry.csv` is tracked at repository root and deliberately contains only `exp,group,run_id`.
- `scripts/experiment_launcher.py train ...` writes the generated bare run id before the job is launched; it also sets `WANDB_TAGS=expNN` so later cloud lookup is unambiguous.
- `scripts/experiment_launcher.py play <exp> <group> ...` reads the CSV. For an empty id it queries the W&B `smp` project once using the `expNN` tag plus a bounded `groupNN` name match, writes the newest matching id into the CSV, then replays it.
- `scripts/experiment_launcher.py sync exp12` backfilled all Exp12 groups 1-18 successfully on 2026-07-23. The table now contains ids for all existing Exp12 groups.

## Play Viewer Diagnosis
- No tracked pre-existing play script was modified in the current worktree. The newly added Exp11/12/13 wrappers explicitly pass `--viewer auto`, which is the upstream CLI default.
- Upstream `mjlab.scripts.play` resolves `--viewer auto` to `native` whenever `DISPLAY` or `WAYLAND_DISPLAY` is non-empty; otherwise it selects Viser. On a remote shell with a stale or nonfunctional display variable, native selection can yield no usable visible viewer. Passing `--viewer viser` forces the browser-based simulator view.
- The current Exp12 scripts have both obsolete `groups1_12` and current `groups1_18` train/play versions. They must be consolidated to comply with the one-current-script-per-experiment rule.
- Consolidated on 2026-07-23: deleted Exp12 `groups1_12` train/play scripts and replaced the split Exp6 group7-12/group13-14 policy scripts with `run_exp6_groups7_14.sh` and `play_exp6_groups7_14_wandb.sh`.
- All retained canonical experiment play scripts now force `--viewer viser`; this makes `bash scripts/play_*.sh ...` use the remote browser viewer regardless of an inherited `DISPLAY`/`WAYLAND_DISPLAY`. They still record video with `--video True`.

## Exp13 Tiptoe Fine-Tuning
- Exp13 G4/G5/G6 all inherit Exp10 group15 command, prior, and static reward, but their moving reward mixes respectively are `(0.5,0.25,0.25)`, `(0.6,0.2,0.2)`, and `(0.7,0.15,0.15)`.
- Exp11 G20 is not foot-tilt-only. Its added terms are `-0.1*support_foot_tilt_penalty` and `-0.4*persistent_single_support_penalty`; its `double_air` weight is zero. The Exp11 base config also changes the moving reward, but that part must not be imported into Exp13 G4/G6 fine-tunes. Exp13 G7/G11/G15 retain only those regularizer weights while preserving each respective parent's moving reward.
- Exp13 G8/G12/G16 use only `-0.05*support_foot_tilt`; G9/G13/G17 use `-0.2`; G10/G14/G18 use `-0.4`. All 12 groups fine-tune their parent G4/G5/G6 model_9999.pt for 3000 iterations.
- `persistent_single_support_penalty` is active for command norm `>0.2` only when exactly one foot has nonzero continuous contact time. It subtracts `0.4 * clamp(max(current_contact_time)-0.45, 0, 0.8)` in G20. It uses the support foot's total uninterrupted contact age, not a separately reset one-foot-phase timer, so preceding double-support time is included. This can penalize legitimate slow walking and must be treated as a gait cadence bias rather than a general anti-tiptoe term.

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
- The current generic ONNX export script is `scripts/export_onnx.sh`.
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
- `scripts/clip_csv_viewer.py` loads raw CSV using `input_fps` and passes it to `CsvMotionLoader(input_fps=..., output_fps=...)`.
- `CsvMotionLoader` computes raw duration as `(input_frames - 1) / input_fps`, then interpolates poses to `output_fps` for visualization.
- The viewer frame slider uses interpolated `motion.output_frames`, and status time is displayed as `frame / output_fps`.
- Playback loop advances one interpolated frame every `1 / playback_fps`; default `playback_fps=50`.
- Therefore default viewer playback preserves real-time speed for 30fps source data by converting 30fps input to 50fps visualization and playing at 50fps.
- Original CSV real duration in seconds is `(num_rows - 1) / input_fps`; a quick approximation is `num_rows / input_fps`, but the code uses the `num_rows - 1` convention.
- Selection export maps visual frames back to original CSV rows by converting visual frame to time with `output_fps`, then time to row index with `input_fps`.

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
- Current direct bash runner for groups 7-14: `scripts/run_exp6_groups7_14.sh`; groups 7-12 use `pretrained_forward_stop.pt`, and groups 13-14 use `forward_stop_all.pt`.
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

## my-dev Foot-Regularized Body Velocity
- `my-dev` contains a related foot/contact regularized body-velocity task: `Smp-BodyVelocity-FootRegularized-G1`.
- Registration is in `my-dev:src/smp/rl/tasks/steering/__init__.py`; env builder is `g1_body_velocity_foot_regularized_smp_env_cfg` in `my-dev:src/smp/rl/tasks/steering/body_velocity_env_cfg.py`.
- It adds a `feet_ground_contact` sensor and rewards: `feet_air_time`, `feet_slide`, `support_foot_tilt`, `persistent_single_support`, `soft_landing`, and `action_rate`.
- This task is not an Exp10 group14/15-specific task. It uses my-dev's body-velocity command range and does not include Exp10 C1 zero-command sampling or Exp10 static-switch reward.
- Training from Exp10 group14/15 checkpoints is therefore a transfer/fine-tune into the foot-regularized task, not a continuation of the exact same Exp10 environment.
- Current branch does not register `Smp-BodyVelocity-FootRegularized-G1`; run the command on `my-dev` or port that task into the current branch first.
- For Exp10 group14/15 fine-tuning, do not discard the original task reward. The conservative design is to keep the successful Exp10 group14/15 command/static-switch reward and add foot regularization terms on top.
- Using only foot regularization rewards would optimize foot appearance/contact behavior without preserving velocity tracking, static stability, or SMP style, so it is high-risk for catastrophic drift.
- my-dev foot regularization is additive: it starts from `g1_body_velocity_smp_env_cfg`, keeps `task_smp_product`, and appends foot/contact/action rewards.
- my-dev foot-regularized command is not Exp10 C1: `x=[-1,2]`, `y=[-1,1]`, `yaw=[-1,1]`, resampling `(3,8)`, no Exp10 zero-command probability/static-switch reward.

## Experiment 11 Body-Velocity Foot Tilt
- Goal: continue from the successful Exp10 body-velocity group14/group15 policies and reduce tiptoe by adding only `support_foot_tilt`.
- Implementation keeps Exp10 group14/group15 command, prior, and `task_smp_product` reward unchanged, then adds one top-level reward term: `support_foot_tilt`.
- No actor or critic observation terms are added for foot contact. The contact sensor exists only in `cfg.scene.sensors` so the simulated reward function can read contact force.
- Added reward formula:
  - `R_total = r_task_smp_product + w_foot_tilt * r_foot_tilt`.
  - `r_foot_tilt = sum_i 1[||F_i|| > 1.0] * ||up_i_xy||^2` over `left_ankle_roll_link` and `right_ankle_roll_link`.
  - `up_i` is the foot body's local z-axis rotated into world frame; flat support feet have low xy magnitude.
- Group mapping:
  - Exp11 group1-4 inherit Exp10 group14 and use weights `-0.05,-0.1,-0.2,-0.3`.
  - Exp11 group5-8 inherit Exp10 group15 and use weights `-0.05,-0.1,-0.2,-0.3`.
- Exp11 group9-26 extend the same experiment with the new moving reward `0.6*r_l*r_y + 0.2*r_l + 0.2*r_y`.
  - group9-17 inherit Exp10 group14; group18-26 inherit Exp10 group15.
  - support-foot-tilt is fixed at `-0.1`.
  - groups 9-11 / 18-20 add gait rhythm only: persistent-single-support weights `-0.1,-0.2,-0.4`.
  - groups 12-14 / 21-23 add drop/no-support only: double-air weights `-0.1,-0.3,-0.6`.
  - groups 15-17 / 24-26 add both: `(-0.4,-0.1)`, `(-0.2,-0.3)`, `(-0.1,-0.6)` for `(gait, drop)`.
- Because `mjlab.scripts.train` cannot resume from an arbitrary local checkpoint path directly, the new training script resumes from W&B via `--agent.resume=True --wandb-run-path ... --wandb-checkpoint-name model_9999.pt`.

## Experiment 12 Theme Priors
- Goal: test whether male/female/children walking styles can be induced by changing the SMP prior.
- Local theme CSVs found:
  - `datasets/csv/theme/walk_male.csv`
  - `datasets/csv/theme/walk_female.csv`
  - `datasets/csv/theme/walk_children.csv`
  - `datasets/csv/theme/stop_static.csv`
- Exp12 group1-12 mapping is theme-major then weight-minor:
  - group1-4: male theme prior with support-foot-tilt weights `-0.05,-0.1,-0.2,-0.3`.
  - group5-8: female theme prior with support-foot-tilt weights `-0.05,-0.1,-0.2,-0.3`.
  - group9-12: children theme prior with support-foot-tilt weights `-0.05,-0.1,-0.2,-0.3`.
- Exp12 group1-12 inherit Exp10 group14:
  - command C1: `P(command=0)=0.3`, otherwise `x,y~U(-2,2)`, `yaw~U(-1,1)`.
  - reward D stop branch: stopping `r_root_stop*r_joint_vel`, then SMP product wrapper.
  - moving branch changed to `0.6*r_l*r_y + 0.2*r_l + 0.2*r_y`.
  - `support_foot_tilt` is added as a separate top-level term: `R_total = r_exp10_group14_task_smp_product + w*r_foot_tilt`.
  - `persistent_single_support` is added with weight `-0.2`, max support time `0.45s`, and active command threshold `0.2`.
  - `double_air` is added with weight `-0.3` and active command threshold `0.2`.
  - only `init_smp_state.params.ckpt_path` and foot-tilt weight change.
- Exp12 group13-18 are prior-only checks:
  - group13-15 inherit Exp10 group14 and replace prior with male/female/children.
  - group16-18 inherit Exp10 group15 and replace prior with male/female/children.
  - no foot-tilt, gait, double-air, or moving-reward override is added.
  - SMP reward follows the style prior because `init_smp_state.ckpt_path` loads `_smp_bundle`, and `smp_guidance_reward` evaluates that loaded bundle.
- Theme command ranges computed by `scripts/analyze_theme_command_ranges.py` from 30fps CSV interpolated to 50fps:
  - male: `x=[-0.4697,1.0390]`, `y=[-0.3238,0.6201]`, `yaw=[-3.2597,4.1955]`.
  - female: `x=[-0.5441,1.0136]`, `y=[-0.2466,0.5320]`, `yaw=[-2.1310,4.5602]`.
  - children: `x=[-0.1635,1.7344]`, `y=[-0.6071,0.4886]`, `yaw=[-2.4046,5.1516]`.
- Exp12 play script follows the standard one-group W&B playback format:
  - `bash scripts/play_exp12_groups1_18_wandb.sh [--gpu N] [--num-envs N] [--video-length N] <group_number> <wandb_run_path>`.
  - It runs `Smp-BodyVelocity-Exp12-Group<group>-G1` with `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, `--video True`, and default `--video-length 1500`.
- Exp12 one-group training script is now `scripts/run_exp12_theme_policy_groups1_18.sh <group> <gpu>`.
- Style matching must treat `male` as a token, not substring, so `walk_female.csv` is not included in male. Scripts use `_`/`-`/`.` token boundaries.

## Experiment 13 Body-Velocity Moving Reward Mix
- Goal: improve single-axis `x/y/yaw` command tracking by replacing the moving reward branch.
- Shared command/prior/SMP/observations come from Exp10 group14 or group15 unchanged.
- Moving reward formula: `r_move = k1*r_l*r_y + k2*r_l + k3*r_y`, with `r_l=exp(-2*||v_xy_body-v_xy_cmd||^2)` and `r_y=exp(-1*(yaw_rate_body-yaw_cmd)^2)`.
- Group1-3 inherit Exp10 group14 and sweep `(k1,k2,k3)` as `(0.5,0.25,0.25)`, `(0.6,0.2,0.2)`, `(0.7,0.15,0.15)`.
- Group4-6 inherit Exp10 group15 with the same sweep.
- Exp13 G19-G21 each retain G4/G5/G6 respectively and add only `-0.05*support_foot_tilt_penalty`, whose per-foot mask is terrain net force above 80N. There is no static under-loaded-foot force reward.
- Scripts: `scripts/run_exp13_groups1_21.sh` and `scripts/play_exp13_groups1_21_wandb.sh`.
- Generic policy export script: `scripts/export_onnx.sh`; it exports any policy checkpoint to a same-directory `.onnx` file. The separate `scripts/export_onnx_with_deadzone.sh` is only for body-velocity configs trained with a command dead zone.

## Experiment 14 Planned Static Flat-Foot And Gait-Duty Ablations
- Exact my-dev raw per-foot tilt implementation is `support_foot_tilt_penalty` in `my-dev:src/smp/rl/tasks/steering/mdp/rewards.py`: it rotates local foot up vector `e_z=[0,0,1]` by the foot world quaternion, computes `t_i=u_{i,x}^2+u_{i,y}^2`, then sums contact-masked feet. Thus `t_i=sin^2(theta_i)`, where `theta_i` is the sole-up-vector angle from world vertical; `t_i=0` when flat and approaches 1 at 90 degrees. It is a raw penalty, not an exponential reward.
- For Exp14 static flat-foot reward, compute the two feet separately. Additive form should use raw `t_L,t_R`; multiplicative form should convert them to `r_tilt_i=exp(-k*t_i)`. Previous additive weights `0.05/0.1/0.2` cannot be reused unchanged as exponential `k` values because their effects differ by orders of magnitude; recommended initial multiplicative grid is `k=1,2,4`.
- A windowed ratio `C_L/(C_L+C_R)` with `C_i=sum_j 1[F_i(j)>F_threshold]` measures left/right contact-duty balance. `|C_L/(C_L+C_R)-0.5|` is zero for balanced support, including double support; it does not constrain cycle duration or step frequency. Gate it to moving commands so static double support is not penalized.
- Current contact sensor infrastructure has `track_air_time=True` and gives continuous `current_contact_time`, but no fixed rolling left/right contact-window accumulator. A true fixed-window duty ratio needs a small per-environment history/state buffer in the reward/command-side implementation; do not misuse `persistent_single_support_penalty`, which only reads continuous current contact duration.
- Exp14 uses its own `exp14_feet_ground_contact` net-force sensor and a reset-aware per-environment ring buffer in `body_velocity/mdp/rewards.py`. The buffer stores 3 seconds of left/right boolean contact flags with `F>1N`; it resets independently when an environment episode resets. It is used by exactly one gait reward term per task, so the history advances once per simulation step.
- Post-Exp14 support-foot investigation: the lowest-change next method is a reward-only support mask `support_i = (F_i > 1N) AND (current_contact_time_i > tau_contact)`, with `tau_contact` around 0.04-0.08s. It keeps a lightly loaded tiptoeing foot eligible after debounce, excludes one-frame touchdown/toe-off noise, and applies flat-foot tilt to both feet during real double support. It needs only a contact sensor with `fields=("found", "force")` and `track_air_time=True`; no actor/critic observation or ONNX input changes.
- Other reward-only candidates: add low foot tangential speed to the support mask; use a soft contact gate based on force instead of a hard threshold; or use a small contact-state FSM (swing/touchdown/support/toe-off) driven by contact rise/fall and timer. Do not use external phase as a reward-only hidden variable because it creates conflicting actor targets; both TienKung and unitree_rl_mjlab expose their phase to the actor when they use phase schedules.

## Experiment Launch Automation
- `mjlab.scripts.train` exposes W&B project/name/tags but no CLI field for a new W&B run id.
- W&B SDK accepts `WANDB_RUN_ID` as an environment setting, so a launcher can generate an id before training and therefore know the complete `<entity>/smp/<id>` playback path without waiting for the website.
- This is distinct from Exp11's `--wandb-run-path`, which identifies the old Exp10 checkpoint to download before fine-tuning; it must remain unchanged.
- Local training output already stores downloaded W&B checkpoints below `logs/rsl_rl/<experiment>/wandb_checkpoints/<run_id>/`, confirming that W&B's opaque run id is the only missing element in the usual playback path.
- `tmux 3.2a` is available. One `smp-experiments` session can contain one named window per training run, so VS Code SSH needs only one attached tmux session while multiple GPU jobs remain individually visible.

## Experiment 15 Design Draft (2026-08-05)
- Exp15 is fixed to `Exp13 G4` only: preserve its prior, command, observation, SMP wrapper, and moving reward mix; do not inherit Exp14 or Exp13 G5/G6 models.
- Draft uses a staged design: Phase 0 controls continued fine-tuning drift, Phase 1 adds only static double-foot tilt, and Phase 2 adds only moving contact-gated support-foot tilt after selecting a viable Phase 1 parent.
- Draft group range is G1-G13: G1-G2 no-new-reward controls; G3-G8 static tilt with `w={0.05,0.1,0.2}` and 1000/3000 iterations; G9-G13 moving-stage control plus low-strength debounced support tilt.
- Moving support mask is reward-only: non-static command, `F_i>1N`, and continuous contact longer than `0.06s`; it does not change actor/critic dimensions or deployment input.
- High-frequency action/joint metrics are diagnostic-only in the draft; no action-smoothing reward is added before confirming the simulator signal.
- User approved the design. Exp15 task/config, diagnostics, train/play scripts, launcher support, and registry rows are now implemented; cached `torch/mjlab` verification passed, but no GPU training smoke test was run.
- Exp15 diagnostics now resolve action indices from the runtime `joint_pos`
  target names and separately monitor 12 leg joints and 4 ankle joints.
  Recorded signals are first difference, second-difference proxy, consecutive
  sign-flip rate, and matching group joint-velocity RMS/peak. The values are
  cached once per policy step and shared by W&B metrics and the CSV recorder;
  they remain diagnostic-only.
