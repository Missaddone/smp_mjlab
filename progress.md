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
- Last planning refresh: 2026-07-02 03:41:53 UTC.
- Count user messages after this refresh and update these files again after 6.
- Counter since last refresh: 0.

## Next Likely Action
- User will choose between play implementation options. If they choose exact
  run-based playback, implement `scripts/play_from_run.py` and then generate
  exact play commands for experiment 1 groups.
