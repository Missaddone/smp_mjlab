# Experiment 12: Body-Velocity Theme Priors

## Goal

Test whether changing the SMP prior can produce different walking styles while keeping the successful Experiment 10 group14 policy configuration and sweeping the same four support-foot-tilt weights used in Exp11.

The theme CSVs live under `datasets/csv/theme`.

## Shared Policy Config

All Exp12 policy groups inherit `Smp-BodyVelocity-Exp10-Group14-G1`:

- command: Exp10 body-velocity C1, `P(command=0)=0.3`, otherwise `x,y~U(-2,2)`, `yaw~U(-1,1)`.
- reward: Exp10 group14 stop-product switch plus one additive `support_foot_tilt` term.
- policy training: `4096 envs`, `10000` iterations, single GPU, W&B logging.

Only the theme prior and `support_foot_tilt` weight change across groups.

## Prior Data

Each style prior uses the matching walk CSV plus common stop/static/stand CSVs:

| Theme | Default staged CSVs | Prior checkpoint |
|---|---|---|
| male | `*male*.csv` + `*stop*/*static*/*stand*.csv` | `datasets/pretrain_ckpt/exp12_theme_male.pt` |
| female | `*female*.csv` + `*stop*/*static*/*stand*.csv` | `datasets/pretrain_ckpt/exp12_theme_female.pt` |
| children | `*children*.csv` + `*stop*/*static*/*stand*.csv` | `datasets/pretrain_ckpt/exp12_theme_children.pt` |

## Policy Groups

```text
R_total = r_exp10_group14_task_smp_product + w_foot_tilt * r_foot_tilt
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

Train one policy group:

```bash
bash scripts/run_exp12_theme_policy_groups1_12.sh 1 0
bash scripts/run_exp12_theme_policy_groups1_12.sh 5 1
bash scripts/run_exp12_theme_policy_groups1_12.sh 9 2
```
