# Experiment 16: Motebu And Static Prior

## Goal

只改变动作数据来源，测试新的全向走路数据能否在不增加平脚奖励的情况下减轻踮脚。
三组 policy 都从零训练，不 resume Exp13 actor。

## Prior Data

prior staging 必须且只能包含四个 CSV：

1. `datasets/csv/theme/motebu.csv`
2. `motebu_mirror.csv`
3. `datasets/csv/theme/stop_static.csv`
4. `stop_static_mirror.csv`

CSV 使用 `base_pos(3) + base_quat_xyzw(4) + joint_pos(29)`，镜像脚本按 `xyzw`
处理四元数。预训练复用 `datasets/norm_stats.npz` 的全量 LAFAN q01/q99，输出：

```text
datasets/pretrain_ckpt/exp16_motebu_stop_static.pt
```

## Shared Policy Configuration

- task family: body-velocity `[v_x, v_y, yaw_rate]`。
- command: `x,y in [-1.5,1.5]`, `yaw in [-2,2]`。
- `P(command=0)=0.3`，与 Exp13 G4-G6 的原配置一致。
- static branch、observation、SMP wrapper 均继承 Exp13 G4-G6。
- environment count: 4096。
- training iterations: 10000。
- logger: W&B project `smp`。

## Groups

| Group | Moving task reward |
|---:|---|
| 1 | $0.5r_l r_y+0.25r_l+0.25r_y$ |
| 2 | $0.6r_l r_y+0.2r_l+0.2r_y$ |
| 3 | $0.7r_l r_y+0.15r_l+0.15r_y$ |

没有增加 foot tilt 或其他额外平脚 reward，因此该实验可以隔离观察 prior 数据本身
对踮脚的影响。

## Commands

```bash
bash scripts/run_exp16_prepare_motebu_stop_static_prior.sh --gpu <gpu>
bash scripts/run_exp16_groups1_3.sh <group> <gpu>
```

```bash
bash scripts/play_exp16_groups1_3_wandb.sh \
  --gpu <gpu> <group> <wandb_run_path>
```

`wandb_run_registry.csv` 已预留 Exp16 G1-G3 三行；也可以通过
`scripts/experiment_launcher.py` 训练或播放 Exp16。
