# Findings

## Clean Task Baseline
- Task: `Smp-BodyVelocity-G1`.
- Prior checkpoint: `datasets/pretrain_ckpt/amp_loco_clips2_mirrored_lafan_norm_12000.pt`.
- This combination supersedes earlier speculation about `Smp-BodyForwardBackward-G1`.

## Notion
- User message log page: `smp_mjlab Codex 用户消息记录`, page id `38f79c70-e2a7-81d2-b84d-f9e18cace163`.
- Experiment flow database: `流程记录`, database id `5972f2d8-d61e-4837-81fd-9f81f383ba72`.
- Flow data source: `collection://f2d5dc66-d42f-45cc-af2e-513bd60aa725`.
- Flow columns: `编号`, `git的commit编号`, `代码改动`, `效果`, `需要达成的目的`, `max-iteration`, `num-envs`, `task-name`, `command设计`, `reward设计`, `速度区间`, `prior来源`.

## Pending Code Facts
- Subagent confirmed exact current `Smp-BodyVelocity-G1` reward formula.
- Current `Smp-BodyVelocity-G1` command comes from `src/smp/rl/tasks/steering/body_velocity_env_cfg.py`; current default is x `[-1, 2]`, y `[-1, 1]`, yaw `[-1, 1]`. Experiment 2 will override x to `[-1, 1]`.
- Current reward function is `body_velocity_task_smp_product` in `src/smp/rl/tasks/steering/mdp/rewards.py`.
- Current reward parameters: `lin_vel_err_scale=2.0`, `yaw_rate_err_scale=1.0`, `lin_vel_weight=0.75`, `yaw_rate_weight=0.25`, `style_floor=0.0`, `fixed_timesteps=(8, 15, 22)`, `ws=6.0`.
- Current formula:
  - `r_lin = exp(-2.0 * sum((root_lin_vel_b_xy - command_lin_vel_b_xy)^2))`
  - `r_yaw = exp(-1.0 * (command_yaw_rate - root_yaw_rate)^2)`
  - `r_task = 0.75 * r_lin + 0.25 * r_yaw`
  - `r_style = smp_guidance_reward(fixed_timesteps=(8,15,22), ws=6.0)`
  - With `style_floor=0.0`, final `r = r_task * r_style`.
- Static foot/root velocity reward is not present as a ready-made function. Root velocity helpers can be reused; foot velocity can read `asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids]`. If contact-gated foot velocity is needed, default `Smp-BodyVelocity-G1` lacks the foot contact sensor.
- Subagent confirmed `Smp-BodyVelocity-G1` currently has no explicit static/standing command branch. `BodyVelocityCommand._resample_command()` samples x/y/yaw continuously with `uniform_`, so even if 0 is inside the interval it will almost never exactly produce `[0, 0, 0]`.
- For experiment 2 static reward switching, existing command design is not enough for a clean static condition. Need either an explicit static sampling probability that sets command to `[0,0,0]`, or an explicit static command flag/mode in the command term.
- Related zero-command facts:
  - `Smp-BodyForwardBackward-G1` fixes y and yaw to 0 but still samples x, so command is `[v_x, 0, 0]`, not static.
  - `Smp-Upstairs-G1` has `rel_standing_envs=0.0`, so standing/static branch is disabled there.
  - There is a GUI `Zero speed` override for steering visualization/debug, but it is not the training sampling logic.

## Git Baseline
- Latest commit before experiment 2 design/code changes: `10a6a41` (`Add body velocity sum task`).
- Per user rule, this is the commit value to write into `流程记录.git的commit编号` for experiment 2 unless a newer commit appears before implementation starts.

## Experiment 2 Implementation
- Added `BodyVelocityCommandCfg.static_prob` with validation in `src/smp/rl/tasks/steering/mdp/commands.py`.
- `BodyVelocityCommand._resample_command()` now sets command exactly to `[0, 0, 0]` for sampled static envs.
- Added reward helpers in `src/smp/rl/tasks/steering/mdp/rewards.py`:
  - `body_velocity_linear_yaw_task_reward`
  - `body_velocity_static_task_reward`
  - `body_velocity_exp2_task_smp_product`
  - `body_velocity_static_exp2_task_smp_product`
- Added two task configs in `src/smp/rl/tasks/steering/body_velocity_env_cfg.py`:
  - `Smp-BodyVelocity-Exp2-G1`: static command probability with non-static linear/yaw product formula.
  - `Smp-BodyVelocity-StaticExp2-G1`: same command, static reward branch for static command.
- Registered both tasks in `src/smp/rl/tasks/steering/__init__.py`.
- Added unit tests in `tests/test_body_velocity_exp2.py`.
- Verified:
  - `.venv/bin/python -m unittest tests/test_body_velocity_exp2.py`
  - `.venv/bin/ruff check ...`
  - train help parses for both new tasks
  - task configs load with expected prior, command ranges, `static_prob=0.2`, and reward params
