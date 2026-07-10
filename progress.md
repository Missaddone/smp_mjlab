# smp_mjlab Progress

## 2026-07-03
- Created planning files after context refresh threshold.
- Logged latest user message to Notion.
- Verified Experiment 4 prior remains `datasets/pretrain_ckpt/pretrained_lafan_run.pt`.
- Added test coverage for `Smp-BodyVelocity-G1` registration, command range, observation split, reward split, and unchanged prior.
- Removed the rejected Python runtime launcher approach for Experiment 4 ablations.
- Added seven Experiment 4 env-cfg files and seven registered task ids.
- Added `scripts/run_exp4_7groups.sh` with direct `CUDA_VISIBLE_DEVICES=N uv run scripts/train.py ...` commands.
- Current user-message counter since this refresh: 4.

## 2026-07-04
- Logged the ONNX export request to Notion.
- Confirmed `my-dev` only has an ONNX/sim2sim note in `notebook.md`; the referenced export script is not present.
- Added `scripts/export_onnx_9999.sh` for exporting `model_9999.pt` to `model_9999.onnx` beside the checkpoint.
- Verified `bash -n scripts/export_onnx_9999.sh` and `bash scripts/export_onnx_9999.sh --help`.
- Refreshed planning context after the message-count threshold.
- Current user-message counter since this refresh: 0.

## 2026-07-06
- Logged the user request about ending Experiment 4 and LAFAN/viser CSV clipping to Notion.
- Recorded Experiment 4 in Notion `流程记录` as item `4`; effect is marked for user follow-up.
- Rechecked `my-dev:scripts/clip_csv_viewer.py` and confirmed CLI parameters and default export path.
- Confirmed current branch is `reconstruct`, worktree is clean, latest commit is `15494f8 formal_exp4,can run train or play or export onnx`.
- Added `scripts/make_static_csv.py` and generated `datasets/csv/forward/stop_static.csv` from `datasets/csv/forward/stop.csv`.
- Could not log the static-stop request to Notion because the Notion auth token expired with HTTP 401.
- Verified static CSV generation and CSV->NPZ conversion using `MPLCONFIGDIR=/tmp/mplconfig_smp_check WARP_CACHE_PATH=/tmp/warp_cache_smp_check`.
- Current user-message counter since this refresh: 1.

## Errors Encountered
| Error | Attempt | Resolution |
| --- | --- | --- |
| `uv: not found` | Tried `uv run scripts/make_static_csv.py` locally | Used `./.venv/bin/python scripts/make_static_csv.py`; final user commands can still use `uv run` on servers with uv. |
| `bash -n` on a Python script | Tried shell syntax check on `scripts/make_static_csv.py` | Replaced with `./.venv/bin/python -m py_compile scripts/make_static_csv.py`. |
| Warp cache write failed under `/home/tyj/.cache/warp` | First CSV->NPZ verification | Set `WARP_CACHE_PATH=/tmp/warp_cache_smp_check` and reran successfully. |
| Notion token expired HTTP 401 | Tried logging user message | Continue work and tell user to re-auth Notion for backfill. |

## 2026-07-07
- Logged Experiment 6 forward command-sampling messages to Notion.
- Added tests for forward Experiment 6 group 4-6 command configs, task registration, original forward preservation, and sampling helper behavior.
- Added `sample_target_speeds` and optional sampling fields to `SteeringCommandCfg`.
- Added and registered `Smp-Forward-Exp6-Group4-G1`, `Smp-Forward-Exp6-Group5-G1`, and `Smp-Forward-Exp6-Group6-G1`.
- Verification passed: `MPLCONFIGDIR=/tmp/mplconfig_smp_tests ./.venv/bin/python -m unittest tests.test_forward_exp6_task tests.test_body_velocity_task`.
- Verification passed: `./.venv/bin/ruff check src/smp/rl/tasks/steering tests/test_forward_exp6_task.py`.
- Current user-message counter since this refresh: 0.
- Logged the Experiment 5 steering-prior request to Notion by inserting a row into the final existing table instead of creating a new table.
- Used the configured project subagent to compare current branch vs `master` for `Smp-Steering-G1`.
- Confirmed current `Smp-Steering-G1` env config and rewards match `master`; only exp6 optional command sampling code differs, with defaults inactive for steering.
- Found batch mirror tooling only in `my-dev:scripts/mirror_motion_data.py`; flagged the CSV quaternion order mismatch risk before recommending bulk use.
- Current user-message counter since this refresh: 1.

## 2026-07-08
- Logged the Experiment 6 group 7-12 request to Notion in the existing final table.
- Added and registered `Smp-Forward-Exp6-Group7-G1` through `Smp-Forward-Exp6-Group12-G1`.
- Added `scripts/run_exp6_groups7_12.sh` with direct `CUDA_VISIBLE_DEVICES=N uv run scripts/train.py ...` commands and W&B logging.
- Updated `tests/test_forward_exp6_task.py` to check group 7-12 command configs and task registration.
- Verification passed: `bash -n scripts/run_exp6_groups7_12.sh`.
- Verification passed: `MPLCONFIGDIR=/tmp/mplconfig_smp_tests ./.venv/bin/python -m unittest tests.test_forward_exp6_task`.
- Verification passed: `./.venv/bin/ruff check src/smp/rl/tasks/steering tests/test_forward_exp6_task.py`.
- Note: `ruff` was not used on the bash script because it treats `.sh` as Python and reports expected syntax errors.
- Current user-message counter since this refresh: 2.
- Logged the Experiment 5 csv_clips prior-comparison request to Notion.
- Added `scripts/run_exp5_prepare_priors.sh` for group1-4 CSV subset assembly, CSV->NPZ conversion, prior pretraining, and checkpoint copy to `datasets/pretrain_ckpt`.
- Added `scripts/run_exp5_policy_groups1_4.sh` for original `Smp-Steering-G1` policy training with the four exp5 prior checkpoints.
- Verified `bash -n scripts/run_exp5_prepare_priors.sh` and `bash -n scripts/run_exp5_policy_groups1_4.sh`.
- Tightened group4 dir-back matching to avoid accidental inclusion of `low_no_dir_back*` and `low_nodir_back*`.
- Current user-message counter since this refresh: 3.
- Logged the correction that all Experiment 5 prior groups must include mirrored counterparts.
- Updated `scripts/run_exp5_prepare_priors.sh` so each group first stages raw CSVs, then runs `scripts/mirror_motion_data.py --include-original --csv --no-npz`, then converts the combined original+mirror directory to NPZ and pretrains.
- Verified `bash -n scripts/run_exp5_prepare_priors.sh`.
- Current user-message counter since this refresh: 4.
- Logged the Experiment 5 W&B play-script request to Notion.
- Added `scripts/play_exp5_groups1_4_wandb.sh` using original `Smp-Steering-G1`, EGL video recording, and four W&B run path environment variables.
- Verified `bash -n scripts/play_exp5_groups1_4_wandb.sh`.
- Current user-message counter since this refresh: 5.
- Logged the Experiment 6 group7-12 positional-argument play-script request to Notion.
- Added `scripts/play_exp6_groups7_12_wandb.sh`; initial version accepted six positional W&B run paths plus optional `--gpu`, `--num-envs`, and `--video-length`.
- Verified `bash -n scripts/play_exp6_groups7_12_wandb.sh` and `bash scripts/play_exp6_groups7_12_wandb.sh --help`.
- Current user-message counter since this refresh: 6. Refresh planning files before/after the next substantial task.
- Logged the correction that Experiment 6 play should run only one requested group at a time.
- Updated `scripts/play_exp6_groups7_12_wandb.sh` to accept exactly `<group_number> <wandb_run_path>` plus optional `--gpu`, `--num-envs`, and `--video-length`.
- Re-verified `bash -n scripts/play_exp6_groups7_12_wandb.sh` and `bash scripts/play_exp6_groups7_12_wandb.sh --help`.
- Logged the Experiment 6 group13/group14 prior-isolation request to Notion.
- Added `src/smp/rl/tasks/steering/forward_exp6_group13_env_cfg.py` and `forward_exp6_group14_env_cfg.py`.
- Registered group13/group14 in `src/smp/rl/tasks/steering/__init__.py`.
- Added `scripts/run_exp6_prepare_forward_stop_all_prior.sh` to mirror `datasets/csv/forward/stop_static.csv`, assemble 8 forward CSV actions, convert to NPZ, pretrain `forward_stop_all`, and copy `datasets/pretrain_ckpt/forward_stop_all.pt`.
- Added `scripts/run_exp6_groups13_14_forward_stop_all.sh` to train group13/group14 with `forward_stop_all.pt`.
- Verification passed: `bash -n scripts/run_exp6_prepare_forward_stop_all_prior.sh scripts/run_exp6_groups13_14_forward_stop_all.sh`.
- Verification passed: `./.venv/bin/ruff check src/smp/rl/tasks/steering/forward_exp6_group13_env_cfg.py src/smp/rl/tasks/steering/forward_exp6_group14_env_cfg.py src/smp/rl/tasks/steering/__init__.py`.
- Verification passed: local Python check printed group13 command as group8 and group14 command as group11.

## Modified Files For Experiment 4
- `src/smp/rl/rewards.py`
- `src/smp/rl/tasks/__init__.py`
- `src/smp/rl/tasks/body_velocity/__init__.py`
- `src/smp/rl/tasks/body_velocity/body_velocity_env_cfg.py`
- `src/smp/rl/tasks/body_velocity/mdp/__init__.py`
- `src/smp/rl/tasks/body_velocity/mdp/commands.py`
- `src/smp/rl/tasks/body_velocity/mdp/rewards.py`
- `tests/test_body_velocity_task.py`
