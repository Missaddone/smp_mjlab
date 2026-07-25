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

## Groups 19-21: 80N Foot-Tilt and Static Double Support

These groups resume G4/G5/G6 respectively for 3000 iterations. Let
`F_i = ||F_net,i||` be each ankle-roll link's terrain net-contact-force
magnitude and let `t_i = ||up_i,xy||^2` be its foot tilt. During every state:

```text
r_foot_tilt80 = sum_i 1[F_i > 80 N] * t_i
```

During a static command, `||[v_x, v_y, yaw_rate]|| < 0.2`, the added support
penalty counts each under-loaded foot independently:

```text
p_static_support80 = sum_i 1[F_i < 80 N]
R = r_task*r_smp - 0.05*r_foot_tilt80 - 0.1*p_static_support80
```

The largest added static support penalty is `-0.2` when both feet are below
80N. This prevents one lightly loaded, tilted foot from bypassing tilt
regularization while keeping the penalty bounded.

| Group | Parent | Added reward terms |
|---:|---:|---|
| 19 | Exp13 G4 | `-0.05*foot_tilt(F_i>80N) - 0.1*static_under80_count` |
| 20 | Exp13 G5 | `-0.05*foot_tilt(F_i>80N) - 0.1*static_under80_count` |
| 21 | Exp13 G6 | `-0.05*foot_tilt(F_i>80N) - 0.1*static_under80_count` |

## Groups 22-27: Static Support Weight Sweep

These groups retain the same per-foot 80N tilt gate and static-only support
definition as G19-G21. They change only the penalty weight for each foot with
`F_i < 80N` during a static command. Every group continues to use
`-0.05*foot_tilt(F_i>80N)` and resumes its listed parent for 3000 iterations.

| Group | Parent | Static under-80N weight per foot |
|---:|---:|---:|
| 22 | Exp13 G4 | `-0.05` |
| 23 | Exp13 G4 | `-0.2` |
| 24 | Exp13 G5 | `-0.05` |
| 25 | Exp13 G5 | `-0.2` |
| 26 | Exp13 G6 | `-0.05` |
| 27 | Exp13 G6 | `-0.2` |

## Commands

Train one policy group:

```bash
bash scripts/run_exp13_groups1_27.sh 1 0
bash scripts/run_exp13_groups1_27.sh 22 0 <parent_g4_wandb_run_path>
```

For W&B path registration and one-command replay, use `experiment_launcher.md` from a tmux window you create yourself:

```bash
uv run scripts/experiment_launcher.py train exp13 <group> --gpu <gpu>
uv run scripts/experiment_launcher.py play exp13 <group> --gpu <gpu> --viewer viser
```

Play one trained policy from W&B:

```bash
bash scripts/play_exp13_groups1_27_wandb.sh --gpu 0 1 <wandb_run_path>
bash scripts/play_exp13_groups1_27_wandb.sh --gpu 0 --viewer viser 27 <wandb_run_path>
```
