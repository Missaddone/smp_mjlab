# Experiment 15: Tracking-Preserving Flat-Foot Fine-Tuning

## Goal

分析并缓解“消除踮脚后，原本的速度跟踪和左右对称性下降”的问题。
本实验只使用 `Exp13 G4` 作为微调起点，不使用 `Exp13 G5/G6`，也不直接继承
`Exp14` 的模型。

## Fixed Baseline

所有策略都从 `Exp13 G4` 的最终策略开始，默认使用其 `model_9999.pt`：

- task：`Smp-BodyVelocity-G1` 的 body-velocity 任务配置。
- prior：沿用 `Exp13 G4`。
- command、observation、SMP reward：全部沿用 `Exp13 G4`。
- moving task reward：

$$
r_{task}^{move}=0.5r_l r_y+0.25r_l+0.25r_y
$$

- 原始 SMP 最终奖励：

$$
R_0=task\_smp\_product=r_{task}r_{smp}
$$

不增加 phase observation，不改变 actor/critic 输入输出维度，不改变部署接口。

## Why Staged Fine-Tuning

Exp14 把静止踮脚和行走支撑脚踮脚同时加入微调，无法判断跟踪下降到底来自：

1. 新奖励本身；
2. 微调轮数和继续训练造成的遗忘；
3. 移动支撑脚判断的离散切换；
4. 仿真中的高频动作输出。

因此 Exp15 分成两个阶段：先只处理静止，再在静止结果中选择可行模型处理行走。

## Phase 0: Fine-Tuning Control

这两个组不增加任何新奖励，只从 `Exp13 G4 model_9999.pt` 继续训练，用来衡量
“继续微调本身”造成的性能变化。

| Group | Parent | Added reward | Iterations | Purpose |
|---:|---|---|---:|---|
| 1 | Exp13 G4 | none | 1000 | 微调漂移对照 |
| 2 | Exp13 G4 | none | 3000 | 微调漂移对照 |

## Phase 1: Static Flat-Foot Fine-Tuning

静止判断只使用已有 command，不新增观测：

$$
m_s=\mathbf{1}\left[\left\|[v_x,v_y,\omega_z]\right\|_2\le 0.2\right]
$$

对左右脚分别计算平脚误差。令脚的世界坐标系局部向上方向为

$$
u_i=R(q_i)e_z
$$

其中 $q_i$ 是第 $i$ 只脚的姿态，$R(q_i)$ 是对应旋转矩阵，
$e_z=[0,0,1]^T$。脚底倾斜量为：

$$
t_i=\left\|(u_i)_ {xy}\right\|_2^2
$$

脚完全水平时 $t_i=0$；倾斜越大，$t_i$ 越大。静止阶段的奖励为：

$$
R=R_0-m_s w_s(t_L+t_R)
$$

| Group | Parent | Static reward addition | Iterations | Purpose |
|---:|---|---|---:|---|
| 3 | Exp13 G4 | $-0.05m_s(t_L+t_R)$ | 1000 | 最小静止平脚约束 |
| 4 | Exp13 G4 | $-0.10m_s(t_L+t_R)$ | 1000 | 中等静止平脚约束 |
| 5 | Exp13 G4 | $-0.20m_s(t_L+t_R)$ | 1000 | 较强静止平脚约束 |
| 6 | Exp13 G4 | $-0.05m_s(t_L+t_R)$ | 3000 | 最小约束的长微调 |
| 7 | Exp13 G4 | $-0.10m_s(t_L+t_R)$ | 3000 | 中等约束的长微调 |
| 8 | Exp13 G4 | $-0.20m_s(t_L+t_R)$ | 3000 | 较强约束的长微调 |

优先比较 G3-G5：如果 1000 轮已经能够稳定双脚平放，优先使用短微调结果，减少
跟踪能力遗忘。G6-G8 只用于判断是否需要更长的静止巩固。

## Phase 2: Moving Support-Foot Flatness

Phase 2 不从 Exp13 G4 直接开始，而是从 Phase 1 中静止稳定、跟踪下降最小的组继续。
具体父组在 Phase 1 测试后确定；在此之前不固定某一个父组。

### Recommended support-foot mask

对脚底接触传感器得到的地面接触力大小：

$$
F_i=\left\|f_i^{terrain}\right\|_2
$$

先用 $1\,N$ 判断是否接触地面，再用连续接触时间 $T_i$ 去抖：

$$
c_i=\mathbf{1}[F_i>1N]\,\mathbf{1}[T_i>0.06s]
$$

只在非静止 command 下启用：

$$
m_m=\mathbf{1}\left[\left\|[v_x,v_y,\omega_z]\right\|_2>0.2\right]
$$

推荐的移动支撑脚惩罚为：

$$
R=R_0-m_m w_m(c_Lt_L+c_Rt_R)
$$

这里不使用 `argmax(F_L,F_R)` 直接选择一只脚。连续接触 mask 能避免左右脚在接触力
接近时频繁切换，降低因奖励不连续造成的步态不对称和跟踪退化。

| Group | Parent | Moving reward addition | Iterations | Purpose |
|---:|---|---|---:|---|
| 9 | Phase 1 best | none | 1000 | Phase 2 微调漂移对照 |
| 10 | Phase 1 best | $-0.02m_m(c_Lt_L+c_Rt_R)$ | 1000 | 极低移动平脚约束 |
| 11 | Phase 1 best | $-0.05m_m(c_Lt_L+c_Rt_R)$ | 1000 | 低强度移动平脚约束 |
| 12 | Phase 1 best | $-0.02m_m(c_Lt_L+c_Rt_R)$ | 3000 | 极低约束的长微调 |
| 13 | Phase 1 best | $-0.05m_m(c_Lt_L+c_Rt_R)$ | 3000 | 低强度约束的长微调 |

Phase 2 的目标不是让飞脚保持水平，而是只约束已经持续接触地面的支撑脚。

## Metrics Only: High-Frequency Behavior

下面的量只记录到 W&B，不进入 reward，也不进入 actor/critic observation：

$$
\Delta a_t=a_t-a_{t-1}
$$

$$
\Delta^2a_t=\Delta a_t-\Delta a_{t-1}
$$

记录：`action_delta_rms`、`action_delta_peak`、`action_accel_proxy_rms`、
`joint_vel_rms`、`joint_vel_peak`、左右脚 tilt、接触比例、速度跟踪误差。

另外单独观察腿部和脚踝的 policy 输出。腿部集合由左右髋、膝、踝共 12 个
关节组成，脚部集合为左右踝 pitch/roll 共 4 个关节。索引在运行时根据
`joint_pos` action term 的 `target_names` 解析，不依赖固定 action 下标。

对每个集合分别记录：

- `*_action_delta_rms/peak`：相邻 policy 输出变化；
- `*_action_accel_proxy_rms/peak`：policy 输出二阶差分；
- `*_action_sign_flip_rate`：连续两次 action 变化方向相反的关节比例。
- `*_joint_vel_rms/peak`：同一组关节实际速度，用于确认 action 抖动是否
  已经传递到仿真中的腿部和脚踝运动。

其中 `leg_*` 对应整条腿，`foot_*` 对应脚踝。二阶差分和方向反转比例同时升高
时，表示腿部或脚踝存在明显的高频交替输出。该方法只需要当前和前两步 action，
不维护长窗口，也不在线执行 FFT。

如果这些指标在 Exp15 明显升高，才单独设计很小的动作平滑奖励；本实验第一阶段
不把平滑项混入，避免同时改变“平脚”和“动作频率”两个因素。

## Prior Mirroring Audit

这不是一个新的训练组，而是 Exp15 的前置审计：

1. 找到生成 `Exp13 G4` prior 的实际远端数据目录或 staging 目录；
2. 确认每个原始动作都有唯一对应的 mirror CSV；
3. 对 mirror 两次的结果与原始 CSV 做数值误差检查；
4. 核对四元数顺序、CSV 列数和 G1 关节左右交换规则。

当前本地代码只能证明镜像脚本按 `base_quat_xyzw` 处理，不能证明历史上生成
`Exp13 G4` prior 时远端确实运行过该脚本。因此不能在没有远端 prior staging 证据时，
声称 G4 的 prior 已完整镜像。

## Implementation

Exp15 已实现为独立任务，不修改原始 `Smp-BodyVelocity-G1`、Exp13 或 Exp14
配置。注册的任务名为：

```text
Smp-BodyVelocity-Exp15-Group1-G1 ... Smp-BodyVelocity-Exp15-Group13-G1
```

训练单组：

```bash
bash scripts/run_exp15_groups1_13.sh <group> <gpu> <source_wandb_run_path> \
  [checkpoint_name] [phase1_static_tilt_weight]
```

G1-G8 的 source 应为 Exp13 G4 的 W&B run；G9-G13 的 source 应为 Phase 1
选出的最佳组。G9-G13 还必须传入父组的静止 tilt 权重，以便继续保留 Phase 1
的静止奖励。例如父组权重是 `0.1`：

```bash
bash scripts/run_exp15_groups1_13.sh 10 0 <phase1_wandb_path> \
  <phase1_checkpoint_name> 0.1
```

默认 checkpoint 是 `model_9999.pt`，如果父组已经继续训练，必须显式传入对应的
`model_*.pt`。

也可以通过 launcher 运行：

```bash
uv run scripts/experiment_launcher.py train exp15 <group> --gpu <gpu>
```

launcher 会自动为 G1-G8 查找 Exp13 G4；G9-G13 必须显式指定：

```bash
uv run scripts/experiment_launcher.py train exp15 <group> --gpu <gpu> \
  --source-wandb-path <phase1_wandb_path> \
  --checkpoint-name <phase1_checkpoint_name> \
  --phase1-static-weight <0.05|0.1|0.2>
```

单组播放：

```bash
bash scripts/play_exp15_groups1_13_wandb.sh --gpu <gpu> <group> <wandb_run_path>
```

### Diagnostic output

每个 Exp15 任务都会创建并持续刷新一个 CSV 文件：

```text
logs/exp15_diagnostics/groupXX_diagnostics.csv
```

文件包含：`step`、有效环境数量、重置环境数量、`action_delta_rms`、
`action_delta_peak`、`action_accel_proxy_rms`、`action_accel_proxy_peak`、
`joint_vel_rms`、`joint_vel_peak`，以及上述 `leg_*`、`foot_*` action 震荡
指标以及对应的 `leg_*`、`foot_*` joint-velocity 指标。训练启动时会在终端打印实际写入路径，
文件每 50 个环境步 flush 一次，环境关闭时再次 flush。

同一批指标也会通过 mjlab 的 metrics manager 进入 W&B 的
`Episode_Metrics/exp15_*` 字段，但它们不参与 reward、observation 或动作计算。

## Decision Gate

推荐顺序：

1. 先运行 G1-G8，确认微调漂移与静止平脚效果；
2. 从 G3-G8 中选择“静止稳定、跟踪下降最小”的一个作为 Phase 2 父策略；
3. 再运行 G9-G13，比较移动支撑脚平脚约束；
4. 根据 W&B 指标和 sim2sim 结果决定是否加入更复杂的 gait 约束。
