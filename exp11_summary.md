# Experiment 11: Body-Velocity Foot Tilt Fine-Tuning

## Goal

Use the successful Experiment 10 body-velocity group14 and group15 policies as
starting points. Groups 1-8 only test foot-tilt penalty. Groups 9-26 switch the
moving reward to the product-mix form and test gait rhythm / drop penalties.

No actor or critic observation is changed. The contact sensor is used only by the simulated reward function.

## Shared Reward Form

Groups 1-8:

```text
R_total = r_task_smp_product + w_foot_tilt * r_foot_tilt
```

`r_task_smp_product` is exactly the original reward from the matching Experiment 10 base group:

- Exp11 group1-4 inherit Exp10 group14.
- Exp11 group5-8 inherit Exp10 group15.

The added foot reward is:

```text
r_foot_tilt = sum_i 1[||F_i|| > 1.0] * ||up_i_xy||^2
```

where `i` is each support foot body, `F_i` is the foot-ground contact force from the MuJoCo contact sensor, and `up_i_xy` is the xy part of the foot body's local z-axis expressed in world frame. A flatter support foot has lower penalty.

Groups 9-26 use the same base command/prior/stop branch, but replace the moving
branch:

```text
r_move = 0.6*r_l*r_y + 0.2*r_l + 0.2*r_y
r_l = exp(-2.0 * ||v_xy_body - v_xy_cmd||^2)
r_y = exp(-1.0 * (yaw_rate_body - yaw_rate_cmd)^2)
```

They also keep `support_foot_tilt` fixed at `-0.10` and sweep:

```text
r_single_support = max(contact_time_single_support - 0.45, 0), clipped at 0.8
r_double_air = 1[both feet are airborne while ||[x,y,yaw]|| > 0.2]
```

## Group Table

| Exp11 group | Base task | Base reward | Added weight / combo |
|---:|---|---|---:|
| 1 | `Smp-BodyVelocity-Exp10-Group14-G1` | Exp10 C1 + stop-product switch | -0.05 |
| 2 | `Smp-BodyVelocity-Exp10-Group14-G1` | Exp10 C1 + stop-product switch | -0.10 |
| 3 | `Smp-BodyVelocity-Exp10-Group14-G1` | Exp10 C1 + stop-product switch | -0.20 |
| 4 | `Smp-BodyVelocity-Exp10-Group14-G1` | Exp10 C1 + stop-product switch | -0.30 |
| 5 | `Smp-BodyVelocity-Exp10-Group15-G1` | Exp10 C1 + stop-product-mix switch | -0.05 |
| 6 | `Smp-BodyVelocity-Exp10-Group15-G1` | Exp10 C1 + stop-product-mix switch | -0.10 |
| 7 | `Smp-BodyVelocity-Exp10-Group15-G1` | Exp10 C1 + stop-product-mix switch | -0.20 |
| 8 | `Smp-BodyVelocity-Exp10-Group15-G1` | Exp10 C1 + stop-product-mix switch | -0.30 |
| 9 | `Smp-BodyVelocity-Exp10-Group14-G1` | moving mix060 + gait only | -0.10 |
| 10 | `Smp-BodyVelocity-Exp10-Group14-G1` | moving mix060 + gait only | -0.20 |
| 11 | `Smp-BodyVelocity-Exp10-Group14-G1` | moving mix060 + gait only | -0.40 |
| 12 | `Smp-BodyVelocity-Exp10-Group14-G1` | moving mix060 + drop only | -0.10 |
| 13 | `Smp-BodyVelocity-Exp10-Group14-G1` | moving mix060 + drop only | -0.30 |
| 14 | `Smp-BodyVelocity-Exp10-Group14-G1` | moving mix060 + drop only | -0.60 |
| 15 | `Smp-BodyVelocity-Exp10-Group14-G1` | moving mix060 + gait/drop | gait -0.40, drop -0.10 |
| 16 | `Smp-BodyVelocity-Exp10-Group14-G1` | moving mix060 + gait/drop | gait -0.20, drop -0.30 |
| 17 | `Smp-BodyVelocity-Exp10-Group14-G1` | moving mix060 + gait/drop | gait -0.10, drop -0.60 |
| 18 | `Smp-BodyVelocity-Exp10-Group15-G1` | moving mix060 + gait only | -0.10 |
| 19 | `Smp-BodyVelocity-Exp10-Group15-G1` | moving mix060 + gait only | -0.20 |
| 20 | `Smp-BodyVelocity-Exp10-Group15-G1` | moving mix060 + gait only | -0.40 |
| 21 | `Smp-BodyVelocity-Exp10-Group15-G1` | moving mix060 + drop only | -0.10 |
| 22 | `Smp-BodyVelocity-Exp10-Group15-G1` | moving mix060 + drop only | -0.30 |
| 23 | `Smp-BodyVelocity-Exp10-Group15-G1` | moving mix060 + drop only | -0.60 |
| 24 | `Smp-BodyVelocity-Exp10-Group15-G1` | moving mix060 + gait/drop | gait -0.40, drop -0.10 |
| 25 | `Smp-BodyVelocity-Exp10-Group15-G1` | moving mix060 + gait/drop | gait -0.20, drop -0.30 |
| 26 | `Smp-BodyVelocity-Exp10-Group15-G1` | moving mix060 + gait/drop | gait -0.10, drop -0.60 |

## Training

Run one group at a time:

```bash
bash scripts/run_exp11_body_velocity_foot_tilt.sh <group> <gpu> [wandb_run_path] [checkpoint_name]
```

Default checkpoint name is `model_9999.pt`.

Default W&B source checkpoints:

- group1-4 and group9-17 resume from `robinbird-harbin-institute-of-technology/smp/xsxw4bwf` (Exp10 group14).
- group5-8 and group18-26 resume from `robinbird-harbin-institute-of-technology/smp/4gvwh984` (Exp10 group15).

So the short form is:

```bash
bash scripts/run_exp11_body_velocity_foot_tilt.sh 1 0
bash scripts/run_exp11_body_velocity_foot_tilt.sh 5 1
```
