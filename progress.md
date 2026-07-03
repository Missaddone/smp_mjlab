# Progress

## 2026-06-30
- Created the first Notion user-message log page.
- Investigated experiment 1 train/play commands from project README and task
  configs.
- Added project subagent workflow for read-only code-structure questions.
- Identified clean task/prior combination requested by user:
  `Smp-BodyVelocity-G1` with
  `amp_loco_clips2_mirrored_lafan_norm_12000.pt`.
- Designed and implemented experiment 2 task/reward variants.
- Created Notion `流程记录` rows `2A` and `2B` with blank `效果` fields.

## 2026-07-02
- Re-created and then refreshed `planning-with-files` context.
- Recovered the original 2026-06-30 09:21:27 UTC play request.
- Provided EGL-fixed play commands for experiment 1 group 1 and group 2.
- Used subagent to clarify `Smp-BodyVelocity-G1` registration, env cfg, reward,
  command, and play config behavior.
- Confirmed current `scripts/play.py` does not restore training run command
  ranges and does not accept `--env.commands...` CLI overrides.
- Presented two implementation options for exact command play:
  - Extend existing `scripts/play.py` with command override flags.
  - Add `scripts/play_from_run.py` to restore command ranges from
    `<run-dir>/params/env.yaml`; this is the recommended option.
- Refreshed planning files at user request and reset the 6-message counter.

## Dialogue Refresh Counter
- Last planning refresh: 2026-07-03 02:35:55 UTC.
- Count user messages after this refresh and update these files again after 6.
- Counter since last refresh: 3.

## Next Likely Action
- User wants to run experiment 2 from branch `tyj-test` while current `master`
  checkout continues running jobs.
- Recommended approach: create/use a separate `git worktree` directory for
  `tyj-test`, then run the two experiment 2 commands from that directory.
- The current `master` checkout should not be switched while jobs are running.
- Added `scripts/run_exp2_tyj_test_worktree.sh` to create/reuse the `tyj-test`
  worktree and then directly run the two experiment 2 train commands.
- The script must not use background jobs, PID tracking, log redirection, or
  wrapper launch functions; keep the train commands explicit.
- If GPUs 0/1 are occupied, run the script with `GPU_IDS="4 5"`; dry-run has
  been checked to emit `CUDA_VISIBLE_DEVICES=4` and `CUDA_VISIBLE_DEVICES=5`.
- User asked for experiment 2 play commands following the experiment 1 group 3
  EGL/video pattern. Reusable play template updated to use `uv run scripts/play.py`,
  `--video True`, and an automatic `find ... model_9999.pt`.
- Subagent found a likely bug in experiment 2 static command sampling: linear
  velocity is assigned to `self.lin_vel_b` before static samples are zeroed.
- User corrected play command standardization: do not use local `find` logic in
  generated commands; provide expected log paths and note if files are absent
  locally.
- Updated experiment 2 reward term keys to use the standard `task_smp_product`
  name in both `g1_body_velocity_exp2_smp_env_cfg` and
  `g1_body_velocity_static_exp2_smp_env_cfg`.
- User requested W&B model-source replacement lines under play commands and
  experiment 3 four-group train commands. Planning refreshed after 6 user
  messages.
- User asked how viewer reset chooses initial robot state. Answer: reset uses
  GSI sampled from the env config's SMP prior checkpoint, not a unified pose and
  not directly the policy checkpoint.

## Latest Code Change
- Updated experiment 2 static reward root velocity scales from `1.5/1.5` to
  `2.0/1.0` in `body_velocity_env_cfg.py` and `rewards.py`.
- Updated `tests/test_body_velocity_exp2.py` so the static reward default
  formula expects the unified `2.0/1.0` root scales.
- Reworked `scripts/run_exp2_tyj_test_worktree.sh` to remove background launch,
  PID files, and log files. Verified with `bash -n` and dry-run.
