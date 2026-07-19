# 实验十配置总结

> 目的：把静止能力从 forward 扩展到 steering 和 body-velocity。

---

## 一、共同 prior

所有 Exp10 policy 训练都使用同一个 prior：

```text
datasets/pretrain_ckpt/exp10_loco_stop_static.pt
```

数据来源：

```text
datasets/csv/loco/*.csv + datasets/csv/forward/stop_static.csv
```

脚本会把 staged CSV 中缺少的 mirror 自动补齐。

生成 prior：

```bash
bash scripts/run_exp10_prepare_prior.sh --gpu 1
```

如果 W&B 初始化慢，可以这样：

```bash
WANDB_INIT_TIMEOUT=300 bash scripts/run_exp10_prepare_prior.sh --gpu 1
```

---

## 二、第一批：Steering 任务

任务名：

```text
Smp-Steering-Exp10-Group{1..12}-G1
```

基础来自当前仓库原始 `Smp-Steering-G1`：

- `rand_tar_dir=True`
- `rand_face_dir=True`
- `resampling_time_range=(3.0, 8.0)`
- 非静止 reward 使用原始 steering task：`0.5*r_vel + 0.5*r_face`
- `r_final = r_task * r_smp`

### Command

| Command | 设计 |
|---|---|
| C1 | `P(command=0)=0.3`，其余 `target_speed ~ U(0, 2)` |
| C2 | `P(command=0)=0.3`，其余 `target_speed ~ U(0, 4)` |
| C3 | C1 + dead zone：采样到 `target_speed < 0.5` 时 command 全部置 0 |
| C4 | C2 + dead zone：采样到 `target_speed < 0.5` 时 command 全部置 0 |

Steering 中 command 全部置 0 指 actor 观测里的 `[target_dir_x, target_dir_y, target_speed, face_dir_x, face_dir_y]` 全部置 0。这个行为只在 C3/C4 的 dead-zone 组中把低速非零 command 变成 0；C1/C2 没有 dead zone，低速非零 command 保持原样。
Reward 的静止判断也使用 `0.5` 阈值：`||target_speed * target_dir|| <= 0.5` 时切到静止 reward。

### Reward

公共定义：

```text
m_still = 1[||v_cmd|| <= 0.5]
r_move = 0.5*r_vel + 0.5*r_face
r_root_stop = exp(-2.0 * ||v_root_xy||^2)
r_joint_vel = exp(-0.02 * sum(|qdot|))
```

Reward C：

```text
r_task = (1-m_still)*r_move
       + m_still*(0.6*r_root_stop + 0.4*r_joint_vel)
```

Reward D：

```text
r_task = (1-m_still)*r_move
       + m_still*(r_root_stop*r_joint_vel)
```

Reward E：

```text
r_task = (1-m_still)*r_move
       + m_still*(0.6*r_root_stop*r_joint_vel + 0.2*r_root_stop + 0.2*r_joint_vel)
```

### Steering 组号

| 组 | Command | Reward |
|---:|---|---|
| 1 | C1 | C |
| 2 | C1 | D |
| 3 | C1 | E |
| 4 | C2 | C |
| 5 | C2 | D |
| 6 | C2 | E |
| 7 | C3 | C |
| 8 | C3 | D |
| 9 | C3 | E |
| 10 | C4 | C |
| 11 | C4 | D |
| 12 | C4 | E |

---

## 三、第二批：Body-Velocity 任务

任务名：

```text
Smp-BodyVelocity-Exp10-Group{13..24}-G1
```

基础来自当前仓库 `Smp-BodyVelocity-G1`：

- actor observation 去掉 `base_lin_vel`
- command 是机器人坐标系下 `[v_x, v_y, yaw_rate]`
- 非静止 reward 使用原始 body-velocity task：`0.75*r_l + 0.25*r_y`
- `r_final = r_task * r_smp`

### Command

| Command | 设计 |
|---|---|
| C1 | `P(command=0)=0.3`，其余 `x,y ~ U(-2,2)`，`yaw ~ U(-1,1)` |
| C2 | `P(command=0)=0.3`，其余 `x,y ~ U(-4,4)`，`yaw ~ U(-2,2)` |
| C3 | C1 + dead zone：`||[x,y,yaw]|| < 0.5` 时 `[x,y,yaw]` 全部置 0 |
| C4 | C2 + dead zone：`||[x,y,yaw]|| < 0.5` 时 `[x,y,yaw]` 全部置 0 |

Body-velocity 的 dead zone 和 reward 静止判断都按完整 command 向量处理：

```text
||[v_x, v_y, yaw_rate]|| < 0.5
```

命中后 command 会被强制改成 `[0, 0, 0]`，reward 也切到静止分支。
C1/C2 没有 dead zone，低速非零 command 保持原样；只有显式 `0.3` 概率采样到的 zero command 会是 `[0,0,0]`。

### Reward

公共定义：

```text
m_still = 1[||[v_x, v_y, yaw_rate]|| < 0.5]
r_move = 0.75*r_l + 0.25*r_y
r_root_stop = exp(-2.0 * ||v_root_xy||^2)
r_joint_vel = exp(-0.02 * sum(|qdot|))
```

Reward C/D/E 与 steering 第一批相同，只是 `r_move` 换成 body-velocity 原始 task。

### Body-Velocity 组号

| 组 | Command | Reward |
|---:|---|---|
| 13 | C1 | C |
| 14 | C1 | D |
| 15 | C1 | E |
| 16 | C2 | C |
| 17 | C2 | D |
| 18 | C2 | E |
| 19 | C3 | C |
| 20 | C3 | D |
| 21 | C3 | E |
| 22 | C4 | C |
| 23 | C4 | D |
| 24 | C4 | E |

---

## 四、运行方式

训练单组：

```bash
bash scripts/run_exp10_policy_groups1_24.sh <group> <gpu>
```

例子：

```bash
bash scripts/run_exp10_policy_groups1_24.sh 1 0
bash scripts/run_exp10_policy_groups1_24.sh 13 1
```

Play 单组：

```bash
bash scripts/play_exp10_groups1_24_wandb.sh --gpu 0 <group> <wandb_run_path>
```

例子：

```bash
bash scripts/play_exp10_groups1_24_wandb.sh --gpu 0 1 <wandb_run_path>
bash scripts/play_exp10_groups1_24_wandb.sh --gpu 1 13 <wandb_run_path>
```
