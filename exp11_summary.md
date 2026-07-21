# Experiment 11: Body-Velocity Foot Tilt Fine-Tuning

## Goal

Use the successful Experiment 10 body-velocity group14 and group15 policies as starting points, then reduce tiptoe behavior by adding only one additive foot-tilt penalty.

No actor or critic observation is changed. The contact sensor is used only by the simulated reward function.

## Shared Reward Form

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

## Group Table

| Exp11 group | Base task | Base reward | Added foot tilt weight |
|---:|---|---|---:|
| 1 | `Smp-BodyVelocity-Exp10-Group14-G1` | Exp10 C1 + stop-product switch | -0.05 |
| 2 | `Smp-BodyVelocity-Exp10-Group14-G1` | Exp10 C1 + stop-product switch | -0.10 |
| 3 | `Smp-BodyVelocity-Exp10-Group14-G1` | Exp10 C1 + stop-product switch | -0.20 |
| 4 | `Smp-BodyVelocity-Exp10-Group14-G1` | Exp10 C1 + stop-product switch | -0.30 |
| 5 | `Smp-BodyVelocity-Exp10-Group15-G1` | Exp10 C1 + stop-product-mix switch | -0.05 |
| 6 | `Smp-BodyVelocity-Exp10-Group15-G1` | Exp10 C1 + stop-product-mix switch | -0.10 |
| 7 | `Smp-BodyVelocity-Exp10-Group15-G1` | Exp10 C1 + stop-product-mix switch | -0.20 |
| 8 | `Smp-BodyVelocity-Exp10-Group15-G1` | Exp10 C1 + stop-product-mix switch | -0.30 |

## Training

Run one group at a time:

```bash
bash scripts/run_exp11_body_velocity_foot_tilt.sh <group> <gpu> [wandb_run_path] [checkpoint_name]
```

Default checkpoint name is `model_9999.pt`.

Default W&B source checkpoints:

- group1-4 resume from `robinbird-harbin-institute-of-technology/smp/xsxw4bwf` (Exp10 group14).
- group5-8 resume from `robinbird-harbin-institute-of-technology/smp/4gvwh984` (Exp10 group15).

So the short form is:

```bash
bash scripts/run_exp11_body_velocity_foot_tilt.sh 1 0
bash scripts/run_exp11_body_velocity_foot_tilt.sh 5 1
```
