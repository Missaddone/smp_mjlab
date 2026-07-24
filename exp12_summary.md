# Experiment 12: Body-Velocity Theme Priors

## Goal

Test whether changing the SMP prior can produce different walking styles.

Exp12 now has two batches:

- group1-12: style prior + foot/gait/drop regularization sweep on top of Exp10 group14.
- group13-18: pure style-prior replacement on top of Exp10 group14/group15; nothing else changes.

The theme CSVs live under `datasets/csv/theme`.

## Shared Policy Config

Exp12 group1-12 inherit `Smp-BodyVelocity-Exp10-Group14-G1`, then update
the moving reward and add foot-contact regularization:

- command: Exp10 body-velocity C1, `P(command=0)=0.3`, otherwise `x,y~U(-2,2)`, `yaw~U(-1,1)`.
- moving reward: `0.6*r_l*r_y + 0.2*r_l + 0.2*r_y`.
- stop reward: unchanged from Exp10 group14, `r_root_stop*r_joint_vel`.
- additive foot rewards:
  - `w_foot_tilt * support_foot_tilt_penalty`.
  - `-0.2 * persistent_single_support_penalty`, active when `||[x,y,yaw]|| > 0.2`.
  - `-0.3 * double_air_penalty`, active when `||[x,y,yaw]|| > 0.2`.
- policy training: `4096 envs`, `10000` iterations, single GPU, W&B logging.

Only the theme prior and `support_foot_tilt` weight change across group1-12.

Exp12 group13-18 inherit the selected Exp10 body-velocity group exactly, then
only replace `cfg.events["init_smp_state"].params["ckpt_path"]` with the theme
prior path. This means the SMP guidance reward also uses the matching theme
prior, because `init_smp_state` loads the model bundle that
`smp_guidance_reward` evaluates.

## Prior Data

Each style prior uses the matching walk CSV plus common stop/static/stand CSVs:

| Theme | Default staged CSVs | Prior checkpoint |
|---|---|---|
| male | `*male*.csv` + `*stop*/*static*/*stand*.csv` | `datasets/pretrain_ckpt/exp12_theme_male.pt` |
| female | `*female*.csv` + `*stop*/*static*/*stand*.csv` | `datasets/pretrain_ckpt/exp12_theme_female.pt` |
| children | `*children*.csv` + `*stop*/*static*/*stand*.csv` | `datasets/pretrain_ckpt/exp12_theme_children.pt` |

## Policy Groups 1-12

```text
r_move = 0.6*r_l*r_y + 0.2*r_l + 0.2*r_y
r_stop = r_root_stop*r_joint_vel
r_task = (1-m_still)*r_move + m_still*r_stop
R_total = r_task*r_smp + w_foot_tilt*r_foot_tilt - 0.2*r_single_support - 0.3*r_double_air
```

| Group | Theme | Foot tilt weight |
|---:|---|---:|
| 1 | male | -0.05 |
| 2 | male | -0.10 |
| 3 | male | -0.20 |
| 4 | male | -0.30 |
| 5 | female | -0.05 |
| 6 | female | -0.10 |
| 7 | female | -0.20 |
| 8 | female | -0.30 |
| 9 | children | -0.05 |
| 10 | children | -0.10 |
| 11 | children | -0.20 |
| 12 | children | -0.30 |

## Policy Groups 13-18

These six groups test the cleaner question: keep the successful Exp10 body-velocity
configuration untouched and only replace the prior. There is no added
`support_foot_tilt`, `persistent_single_support`, `double_air`, or moving-reward
override in these groups.

| Group | Base config | Theme prior | Reward/command change |
|---:|---|---|---|
| 13 | Exp10 group14 | male | prior only |
| 14 | Exp10 group14 | female | prior only |
| 15 | Exp10 group14 | children | prior only |
| 16 | Exp10 group15 | male | prior only |
| 17 | Exp10 group15 | female | prior only |
| 18 | Exp10 group15 | children | prior only |

Current local files match this convention:

- `datasets/csv/theme/walk_male.csv`
- `datasets/csv/theme/walk_female.csv`
- `datasets/csv/theme/walk_children.csv`
- `datasets/csv/theme/stop_static.csv`

## Commands

Analyze command ranges:

```bash
uv run scripts/analyze_theme_command_ranges.py --input-dir datasets/csv/theme
```

Current analyzed ranges from local CSVs:

| Style | Body-frame x range | Body-frame y range | Body yaw-rate range |
|---|---:|---:|---:|
| male | `[-0.4697, 1.0390]` | `[-0.3238, 0.6201]` | `[-3.2597, 4.1955]` |
| female | `[-0.5441, 1.0136]` | `[-0.2466, 0.5320]` | `[-2.1310, 4.5602]` |
| children | `[-0.1635, 1.7344]` | `[-0.6071, 0.4886]` | `[-2.4046, 5.1516]` |

Note: Exp12 intentionally keeps Exp10 group14 command ranges unchanged even though the theme CSV yaw-rate range is wider than `[-1, 1]`.

Prepare the three theme priors:

```bash
bash scripts/run_exp12_prepare_theme_priors.sh --gpu 0
```

Train one policy group from the full 1-18 set:

```bash
bash scripts/run_exp12_theme_policy_groups1_18.sh 13 0
bash scripts/run_exp12_theme_policy_groups1_18.sh 14 1
bash scripts/run_exp12_theme_policy_groups1_18.sh 18 2
```

Play one trained policy from W&B and record video:

```bash
bash scripts/play_exp12_groups1_18_wandb.sh --gpu 0 13 <wandb_run_path>
bash scripts/play_exp12_groups1_18_wandb.sh --gpu 1 --video-length 1500 14 <wandb_run_path>
bash scripts/play_exp12_groups1_18_wandb.sh --gpu 2 --video-length 1500 18 <wandb_run_path>
```

Open Viser and manually control body-velocity command sliders:

```bash
bash scripts/play_exp12_groups1_18_wandb.sh --gpu 0 --viewer viser 13 <wandb_run_path>
```

In the Viser panel, enable the `Body_velocity` command controls, then adjust
`lin_vel_x`, `lin_vel_y`, and `yaw_rate`. The slider ranges are the same as the
group's random command sampling ranges.

The play script uses the same task id as training:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=<gpu> \
  uv run scripts/play.py Smp-BodyVelocity-Exp12-Group<group>-G1 \
    --wandb-run-path <wandb_run_path> \
    --num-envs 1 \
    --video True \
    --video-length 1500 \
    --viewer viser
```
