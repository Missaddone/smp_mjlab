# Experiment 13: Body-Velocity Moving Reward Product Mix

## Goal

Test whether single-axis `x/y/yaw` command tracking improves when the moving
branch of the body-velocity reward uses a product-mix form instead of the
original `0.75*r_l + 0.25*r_y`.

## Shared Config

Each group inherits either Exp10 group14 or Exp10 group15:

- command: Exp10 body-velocity C1, `P(command=0)=0.3`, otherwise `x,y~U(-2,2)`, `yaw~U(-1,1)`.
- prior: unchanged, `datasets/pretrain_ckpt/exp10_loco_stop_static.pt`.
- stop reward: unchanged from the selected Exp10 base group.
- SMP wrapper: unchanged, `R = r_task * r_smp`, `t=(8,15,22)`, `ws=6`.

Only the moving reward changes:

```text
r_move = k1*r_l*r_y + k2*r_l + k3*r_y
r_l = exp(-2.0 * ||v_xy_body - v_xy_cmd||^2)
r_y = exp(-1.0 * (yaw_rate_body - yaw_rate_cmd)^2)
```

## Groups 1-6: From-Scratch Reward Sweep

| Group | Base | Stop reward | k1 | k2 | k3 |
|---:|---|---|---:|---:|---:|
| 1 | Exp10 group14 | `r_root_stop*r_joint_vel` | 0.5 | 0.25 | 0.25 |
| 2 | Exp10 group14 | `r_root_stop*r_joint_vel` | 0.6 | 0.20 | 0.20 |
| 3 | Exp10 group14 | `r_root_stop*r_joint_vel` | 0.7 | 0.15 | 0.15 |
| 4 | Exp10 group15 | `0.6*r_root_stop*r_joint_vel + 0.2*r_root_stop + 0.2*r_joint_vel` | 0.5 | 0.25 | 0.25 |
| 5 | Exp10 group15 | `0.6*r_root_stop*r_joint_vel + 0.2*r_root_stop + 0.2*r_joint_vel` | 0.6 | 0.20 | 0.20 |
| 6 | Exp10 group15 | `0.6*r_root_stop*r_joint_vel + 0.2*r_root_stop + 0.2*r_joint_vel` | 0.7 | 0.15 | 0.15 |

## Groups 7-18: G4-G6 Foot-Regularization Fine-Tuning

These groups resume each parent policy's `model_9999.pt`; they do not start
from scratch. The parent group's command, prior, stop branch, and moving
reward mix remain unchanged. The following terms are added after the original
SMP product reward:

```text
R = r_task * r_smp
  + w_tilt * support_foot_tilt_penalty
  + w_support * persistent_single_support_penalty
```

`support_foot_tilt_penalty` is the sum of the squared horizontal component of
each contacting foot's local up vector. `persistent_single_support_penalty`
is the support-contact time above `0.45 s`, capped at `0.8 s`, only while
`||[v_x, v_y, yaw_rate]|| > 0.2`. Its G20-style parameters are retained
exactly: `w_tilt=-0.1`, `w_support=-0.4`.

| Group | Parent | Added reward terms |
|---:|---:|---|
| 7 | Exp13 G4 | G20 style: `-0.1*foot_tilt - 0.4*persistent_single_support` |
| 8 | Exp13 G4 | `-0.05*foot_tilt` |
| 9 | Exp13 G4 | `-0.2*foot_tilt` |
| 10 | Exp13 G4 | `-0.4*foot_tilt` |
| 11 | Exp13 G5 | G20 style: `-0.1*foot_tilt - 0.4*persistent_single_support` |
| 12 | Exp13 G5 | `-0.05*foot_tilt` |
| 13 | Exp13 G5 | `-0.2*foot_tilt` |
| 14 | Exp13 G5 | `-0.4*foot_tilt` |
| 15 | Exp13 G6 | G20 style: `-0.1*foot_tilt - 0.4*persistent_single_support` |
| 16 | Exp13 G6 | `-0.05*foot_tilt` |
| 17 | Exp13 G6 | `-0.2*foot_tilt` |
| 18 | Exp13 G6 | `-0.4*foot_tilt` |

## Commands

Train one policy group:

```bash
bash scripts/run_exp13_groups1_18.sh 1 0
bash scripts/run_exp13_groups1_18.sh 7 0 <parent_wandb_run_path>
```

For W&B path registration and one-command replay, use `experiment_launcher.md` from a tmux window you create yourself:

```bash
uv run scripts/experiment_launcher.py train exp13 <group> --gpu <gpu>
uv run scripts/experiment_launcher.py play exp13 <group> --gpu <gpu> --viewer viser
```

Play one trained policy from W&B:

```bash
bash scripts/play_exp13_groups1_18_wandb.sh --gpu 0 1 <wandb_run_path>
bash scripts/play_exp13_groups1_18_wandb.sh --gpu 0 --viewer viser 18 <wandb_run_path>
```
