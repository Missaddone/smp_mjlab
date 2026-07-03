# Findings

## Notion
- User message log page: `smp_mjlab Codex 用户消息记录`.
- User message log page id:
  `38f79c70-e2a7-81d2-b84d-f9e18cace163`.
- Experiment flow database: `流程记录`.
- Flow database id: `5972f2d8-d61e-4837-81fd-9f81f383ba72`.
- Flow data source:
  `collection://f2d5dc66-d42f-45cc-af2e-513bd60aa725`.
- Flow columns: `编号`, `git的commit编号`, `代码改动`, `效果`,
  `需要达成的目的`, `max-iteration`, `num-envs`, `task-name`,
  `command设计`, `reward设计`, `速度区间`, `prior来源`.

## Subagent Rule
- For read-only project structure / code logic / relationship questions, use a
  subagent with model `gpt-5.4` and reasoning `medium`.
- Recent subagent findings should generally be trusted, but code can be checked
  locally when command syntax or paths need verification.

## `Smp-BodyVelocity-G1` Task Facts
- Registration is generated in
  `src/smp/rl/tasks/steering/__init__.py`.
- `BodyVelocity` maps to `g1_body_velocity_smp_env_cfg` in
  `src/smp/rl/tasks/steering/body_velocity_env_cfg.py`.
- Train env: `g1_body_velocity_smp_env_cfg(play=False)`.
- Play env: `g1_body_velocity_smp_env_cfg(play=True)`.
- Command implementation:
  `src/smp/rl/tasks/steering/mdp/commands.py`, class
  `BodyVelocityCommand`.
- Reward implementation:
  `src/smp/rl/tasks/steering/mdp/rewards.py`, function
  `body_velocity_task_smp_product`.
- Original reward parameters:
  - `lin_vel_err_scale=2.0`
  - `yaw_rate_err_scale=1.0`
  - `lin_vel_weight=0.75`
  - `yaw_rate_weight=0.25`
  - `style_floor=0.0`
  - `fixed_timesteps=(8, 15, 22)`
  - `ws=6.0`
- Original reward formula:
  - `r_lin = exp(-2.0 * ||v_xy - v'_xy||^2)`
  - `r_yaw = exp(-1.0 * ||v_yaw - v'_yaw||^2)`
  - `r_task = 0.75 * r_lin + 0.25 * r_yaw`
  - `r_style = smp_guidance_reward(t=(8,15,22), ws=6.0)`
  - `r_final = r_task * r_style`

## Experiment 1 Facts
- Correct common config:
  - Task: `Smp-BodyVelocity-G1`
  - Prior: `datasets/pretrain_ckpt/pretrained_loco.pt`
  - Reward: original linear/yaw task reward
  - Base: `4096` envs, `10000` iterations, single GPU
- Group checkpoints:
  - Group 1:
    `logs/rsl_rl/smp_exp1_body_velocity_loco/2026-06-30_02-25-46_range_1_1_1/model_9999.pt`
  - Group 2:
    `logs/rsl_rl/smp_exp1_body_velocity_loco/2026-06-30_02-44-33_range_4_4_2/model_9999.pt`
  - Group 3:
    `logs/rsl_rl/smp_exp1_body_velocity_loco/2026-06-30_12-39-16_range_xneg4_4_y0_yaw0/model_9999.pt`
  - Group 4:
    `logs/rsl_rl/smp_exp1_body_velocity_loco/2026-06-30_12-43-09_range_x0_4_y0_yaw0/model_9999.pt`

## Experiment 2 Facts
- Latest commit before experiment 2 design/code changes was `10a6a41`
  (`Add body velocity sum task`).
- Implemented experiment 2 code touched:
  - `src/smp/rl/tasks/steering/mdp/commands.py`
  - `src/smp/rl/tasks/steering/mdp/rewards.py`
  - `src/smp/rl/tasks/steering/body_velocity_env_cfg.py`
  - `src/smp/rl/tasks/steering/__init__.py`
  - `tests/test_body_velocity_exp2.py`
- Added tasks:
  - `Smp-BodyVelocity-Exp2-G1`
  - `Smp-BodyVelocity-StaticExp2-G1`
- Added explicit static command probability via `static_prob`.
- Added static-specific reward branch for the static reward task.
- Static branch root velocity reward was aligned with moving branch scales:
  `r_l = exp(-2.0 * ||v_xy||^2)` and
  `r_y = exp(-1.0 * v_yaw^2)`. The static-only difference is the additional
  foot velocity product penalty.
- Final reward term key for experiment 2 env configs must remain
  `task_smp_product` for comparability, even though the Python reward functions
  differ (`body_velocity_exp2_task_smp_product` vs
  `body_velocity_static_exp2_task_smp_product`).
- Command config for both experiment 2 tasks is in
  `src/smp/rl/tasks/steering/body_velocity_env_cfg.py`, via
  `_body_velocity_command(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, static_prob=0.2)`.
- Command sampling is in `src/smp/rl/tasks/steering/mdp/commands.py`,
  `BodyVelocityCommand._resample_command`.
- Important issue found by subagent: current static command logic writes
  `self.lin_vel_b[env_ids] = lin_vel_b` before zeroing `lin_vel_b[static_mask]`.
  This means yaw is zeroed for static samples, but linear command may remain the
  sampled value unless the implementation is fixed.
- When generating play commands for remote runs, give the expected standard log
  path such as
  `logs/rsl_rl/<experiment>/<timestamp>_<run-name>/model_9999.pt`. If local logs
  are absent, mention that fact without replacing the standard path with local
  discovery code.
- For W&B-based play, replace the local `--checkpoint-file ...` line with
  `--wandb-run-path <org-or-entity>/<wandb-project>/<wandb-run-id>` as in the
  README.

## Play Reset / GSI
- `scripts/play.py` registers SMP tasks and delegates to `mjlab.scripts.play`.
  The env config includes `init_smp_state` as a startup event and `gsi_reset` as
  a reset event.
- `init_smp_state` loads the frozen SMP denoiser from
  `cfg.events["init_smp_state"].params["ckpt_path"]`, then pre-samples a GSI
  pool of motion windows from that prior.
- On reset, `gsi_reset` uniformly samples windows from `env._smp_gsi_pool`.
  `_prime_sim_and_buffer` writes the last frame to the robot root/joint sim
  state and fills the SMP feature buffer with the whole sampled window.
- Therefore viewer "reset environment" is prior-dependent and stochastic; it is
  not a single unified default pose. It depends on the current registered
  play-env config's `init_smp_state.ckpt_path`, not directly on the policy
  checkpoint path.
- Important for prior-comparison experiments: if train changed prior only via
  CLI override, plain `scripts/play.py <TASK> --checkpoint-file ...` may use the
  task's current default play prior rather than the run's training prior.

## Git Worktree Cleanup
- If `git switch tyj-test` fails because the branch is already checked out in
  another worktree, run `git worktree list`, inspect the listed worktree path,
  then remove it with `git worktree remove <path>` after confirming no training
  job still depends on that directory.
- If the worktree directory was manually deleted, run `git worktree prune`.
- Use `git worktree remove --force <path>` only after saving/stashing any
  uncommitted changes and confirming no process is using it.
- If the current checkout is `master` and must keep running jobs alive, run
  `tyj-test` from a separate `git worktree` directory. Do not switch the active
  checkout used by running jobs.
- Avoid sharing an editable project virtualenv blindly across checkouts because
  imports may resolve to the old checkout. Robust path: run `uv sync --frozen`
  once inside the worktree, then use `uv run` there. Fast path: use the existing
  venv interpreter with `PYTHONPATH=$PWD/src` from the worktree.

## Play Consistency Finding
- `scripts/play.py` is a thin wrapper around `mjlab.scripts.play.main()`.
- `PlayConfig` only supports flat play parameters such as `--checkpoint-file`,
  `--num-envs`, `--video`, `--viewer`, `--device`, and `--log-root`.
- Existing `scripts/play.py` does not support `--env.commands...` overrides.
- Training supports `--env.commands...` because `TrainConfig` contains nested
  `env` and `agent` configs; play does not.
- Current play only loads the current registered `play_env_cfg` and actor
  checkpoint weights. It does not read `params/env.yaml` or `params/agent.yaml`
  from the run directory.
- Therefore, for experiments where command range was changed at train time,
  plain `task + checkpoint` play can mismatch the training command range.
- Recommended fix: add `scripts/play_from_run.py` that restores only
  `commands.steering.lin_vel_x_min/max`, `lin_vel_y_min/max`, and
  `yaw_rate_min/max` from `<run-dir>/params/env.yaml`.

## Server Play Rendering
- Use EGL on the server to avoid
  `mujoco.FatalError: an OpenGL platform library has not been loaded`.
- Required env vars in play commands:
  - `MUJOCO_GL=egl`
  - `PYOPENGL_PLATFORM=egl`
  - `MUJOCO_EGL_DEVICE_ID=0`
  - `MPLCONFIGDIR=/tmp/mplconfig_smp_play`
- `--video` must be followed by `True`: `--video True`.

## Experiment 3 Direct Command Facts
- User corrected experiment 3 forward baseline prior: it must be
  `pretrained_loco.pt`, not `amp_run_clips2_mirrored_lafan_norm_128.pt`.
- User abandoned all prior isolated-workspace/script-launch approaches for
  experiment 3.
- Use direct README-style `uv run scripts/train.py` commands.
- Change only the prior checkpoint between baseline/custom runs for the same
  task. Logging/run-name/GPU assignment are allowed command metadata.
- Use `--agent.max-iterations 10000`, `--env.scene.num-envs 4096`, W&B logger,
  and one GPU per run by default.
