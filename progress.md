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

## Modified Files For Experiment 4
- `src/smp/rl/rewards.py`
- `src/smp/rl/tasks/__init__.py`
- `src/smp/rl/tasks/body_velocity/__init__.py`
- `src/smp/rl/tasks/body_velocity/body_velocity_env_cfg.py`
- `src/smp/rl/tasks/body_velocity/mdp/__init__.py`
- `src/smp/rl/tasks/body_velocity/mdp/commands.py`
- `src/smp/rl/tasks/body_velocity/mdp/rewards.py`
- `tests/test_body_velocity_task.py`
