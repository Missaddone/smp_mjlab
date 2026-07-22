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

## Groups

| Group | Base | Stop reward | k1 | k2 | k3 |
|---:|---|---|---:|---:|---:|
| 1 | Exp10 group14 | `r_root_stop*r_joint_vel` | 0.5 | 0.25 | 0.25 |
| 2 | Exp10 group14 | `r_root_stop*r_joint_vel` | 0.6 | 0.20 | 0.20 |
| 3 | Exp10 group14 | `r_root_stop*r_joint_vel` | 0.7 | 0.15 | 0.15 |
| 4 | Exp10 group15 | `0.6*r_root_stop*r_joint_vel + 0.2*r_root_stop + 0.2*r_joint_vel` | 0.5 | 0.25 | 0.25 |
| 5 | Exp10 group15 | `0.6*r_root_stop*r_joint_vel + 0.2*r_root_stop + 0.2*r_joint_vel` | 0.6 | 0.20 | 0.20 |
| 6 | Exp10 group15 | `0.6*r_root_stop*r_joint_vel + 0.2*r_root_stop + 0.2*r_joint_vel` | 0.7 | 0.15 | 0.15 |

## Commands

Train one policy group:

```bash
bash scripts/run_exp13_body_velocity_moving_reward_mix.sh 1 0
bash scripts/run_exp13_body_velocity_moving_reward_mix.sh 2 1
```

Play one trained policy from W&B:

```bash
bash scripts/play_exp13_groups1_6_wandb.sh --gpu 0 1 <wandb_run_path>
bash scripts/play_exp13_groups1_6_wandb.sh --gpu 0 --viewer viser 2 <wandb_run_path>
```
