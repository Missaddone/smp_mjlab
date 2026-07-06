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
