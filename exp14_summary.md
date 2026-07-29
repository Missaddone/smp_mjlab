# Experiment 14: G5 Flat-Foot, Contact-Duty, And Moving Support-Foot Fine-Tuning

All 16 tasks independently resume Exp13 G5 (`model_9999.pt`). They preserve
its prior, body-velocity command, stop branch, moving reward mix, observation,
and SMP reward. Only the listed additional reward behavior changes.

Let `R0 = r_task * r_smp`. For each foot, its raw tilt is
`t_i = u_i,x^2 + u_i,y^2`, where `u_i` is that foot link's local z-axis in the
world frame. Flat feet have `t_i = 0`.

The duty error uses a rolling 3-second window and a 1N terrain net-force
contact condition:

`d = abs(C_left / (C_left + C_right + 1e-6) - 0.5)`.

It is active only when `||[v_x, v_y, yaw_rate]|| > 0.2`.

| Groups | Parent | Additional reward | Iterations |
|---|---:|---|---:|
| 1, 2 | Exp13 G5 | static `R = R0 - 0.2*(t_left+t_right)` | 3000, 6000 |
| 3, 4 | Exp13 G5 | static `R = R0 - 0.5*(t_left+t_right)` | 3000, 6000 |
| 5, 6 | Exp13 G5 | static `R = R0 - 1.0*(t_left+t_right)` | 3000, 6000 |
| 7, 8 | Exp13 G5 | static `R = R0*exp(-t_left)*exp(-t_right)` | 3000, 6000 |
| 9, 10 | Exp13 G5 | moving `R = R0*exp(-2*d)` | 3000, 6000 |
| 11, 12 | Exp13 G5 | moving `R = R0 - 0.05*d` | 3000, 6000 |
| 13, 14 | Exp13 G5 | moving `R = R0 - 0.1*d` | 3000, 6000 |
| 15, 16 | Exp13 G5 | moving `R = R0 - 0.2*d` | 3000, 6000 |
| 17, 18 | Exp13 G5 | moving `R = R0 - 0.1*sum_i(s_i*t_i)`, debounce 0.04 s | 3000, 6000 |
| 19, 20 | Exp13 G5 | moving `R = R0 - 0.2*sum_i(s_i*t_i)`, debounce 0.04 s | 3000, 6000 |
| 21, 22 | Exp13 G5 | moving `R = R0 - 0.1*sum_i(s_i*t_i)`, debounce 0.06 s | 3000, 6000 |
| 23, 24 | Exp13 G5 | moving `R = R0 - 0.2*sum_i(s_i*t_i)`, debounce 0.06 s | 3000, 6000 |
| 25, 26 | Exp13 G5 | moving `R = R0 - 0.1*t_argmax(F_left,F_right)` | 3000, 6000 |
| 27, 28 | Exp13 G5 | moving `R = R0 - 0.2*t_argmax(F_left,F_right)` | 3000, 6000 |
| 29, 30 | Exp13 G5 | moving `R = R0 - 0.4*t_argmax(F_left,F_right)` | 3000, 6000 |

For G17-G24, `s_i = 1[F_i > 1N] * 1[T_contact,i > tau]`. For G25-G30,
the selected foot must have `max(F_left,F_right) > 1N`; otherwise no movement
tilt penalty is applied. Neither family adds the successful static double-foot
tilt terms from G1-G8.

Train one group directly:

```bash
bash scripts/run_exp14_groups1_30.sh <group> <gpu> \
  robinbird-harbin-institute-of-technology/smp/j8wnqojl
```

Or use the registry-aware launcher, which resolves Exp13 G5 automatically:

```bash
uv run scripts/experiment_launcher.py train exp14 <group> --gpu <gpu>
```

Play one completed group from W&B:

```bash
bash scripts/play_exp14_groups1_30_wandb.sh --gpu <gpu> <group> \
  robinbird-harbin-institute-of-technology/smp/<run_id>
```
