# 实验九结果总结

> 目的：理清实验九各组在对比什么、哪些可以用、基于什么继续改进

---

## 一、实验在对比什么

第一阶段是两个维度的交叉：**4 种 prior × 5 种 reward = 20 组**。

第二阶段在当前最好组合 **Group 19 = P4 + Reward D** 的基础上，补充 **21-33 组**，重点处理静止时上半身高频晃动和低速缺失问题。

### Prior 维度（数据来源逐步增加 stop 相关动作）

| Prior | 数据组合 | 设计意图 |
|-------|---------|---------|
| P1 | forward + walk_to_stop | 最小增量：只加"走→停"切换动作 |
| P2 | forward + walk_to_stop + stop_to_walk | 加上"停→走"切换动作，验证双向切换数据是否有帮助 |
| P3 | forward + walk_to_stop + stop_to_walk + stop_static | 再加纯静止动作，验证静止状态数据是否有帮助 |
| P4 | forward + stop_static | **极简对照**：只加静止动作，不加切换动作，看是否足够 |
| P5 | forward + stop_static + g1_low_walk | 在 P4 基础上补 1m/s 以内低速行走，验证低速数据是否能缓解继电特性 |

**核心问题**：prior 里加什么 stop 相关数据，才最有利于 policy 学会走停切换？

### Reward 维度（静止 reward 从简单到复杂）

| Reward | 公式 | 设计意图 |
|--------|------|---------|
| A | `r_vel - 0.02×m_still×Σ|qdot|` | 最简单：静止时惩罚关节速度（实验七 G4 baseline） |
| B | `r_vel×[(1-m_still)+m_still×exp(-0.02×Σ|qdot|)]` | 门控：静止时用指数衰减压制 r_vel |
| C | `(1-m_still)×r_vel + m_still×(0.6×r_root_stop + 0.4×r_joint_vel)` | 加权和切换：静止时切到根速度惩罚+关节速度惩罚 |
| D | `(1-m_still)×r_vel + m_still×(r_root_stop × r_joint_vel)` | 乘积切换：静止时两项惩罚取乘积，更强 |
| E | `(1-m_still)×r_vel + m_still×(0.6×r_root_stop×r_joint_vel + 0.2×r_root_stop + 0.2×r_joint_vel)` | 乘积混合切换：乘积为主+单项兜底 |
| F | `(1-m_still)×r_vel + m_still×(r_root_stop × r_joint_vel × r_root_ang)` | 在 D 基础上抑制躯干 roll/pitch 角速度 |
| G | `(1-m_still)×r_vel + m_still×(r_root_stop × r_joint_vel × r_upper_joint_vel)` | 在 D 基础上抑制上半身关节速度 |
| H | `(1-m_still)×r_vel + m_still×(r_root_stop × r_joint_vel × r_action_smooth)` | 在 D 基础上抑制连续动作输出跳变 |

**核心问题**：静止时的 reward 怎么设计，才能让 agent 在 speed=0 时停住，同时不破坏正常运动？

---

## 二、20 组结果一览

### 结果标记说明
- **继电** = 有继电特性（缺少低速运动，速度切换不连续，机器人自己震动前进）
- **抖** = 0 速下会抖
- **减抖脚不动** = 0 速下抖动大幅减少，上身会抖，但脚真的没怎么动
- **崩** = 完全不会跑步和/或走路

| 组 | Prior | Reward | 跑步 | 走路 | 静止表现 |
|----|-------|--------|------|------|---------|
| 1 | P1 | A | 正常 | 正常 | 继电，0速抖 |
| 2 | P1 | B | 正常 | 正常 | 继电，0速抖 |
| 3 | P1 | C | 正常 | 正常 | 继电，减抖脚不动 |
| 4 | P1 | D | **不对称** | **崩** | 减抖脚不动 |
| 5 | P1 | E | **崩** | **崩** | 减抖脚不动 |
| 6 | P2 | A | 正常 | 正常 | 继电，0速抖 |
| 7 | P2 | B | **崩** | **崩** | 0速抖 |
| 8 | P2 | C | **崩** | **崩** | 0速抖 |
| 9 | P2 | D | **崩** | **崩** | 0速抖 |
| 10 | P2 | E | **崩** | **崩** | 减抖脚不动 |
| 11 | P3 | A | 待测试 | 待测试 | 待测试 |
| 12 | P3 | B | 待测试 | 待测试 | 待测试 |
| 13 | P3 | C | 待测试 | 待测试 | 待测试 |
| 14 | P3 | D | 待测试 | 待测试 | 待测试 |
| 15 | P3 | E | **崩** | **崩** | 减抖脚不动 |
| 16 | P4 | A | 正常 | 正常 | 继电，0速抖 |
| 17 | P4 | B | — | — | **完全不停** |
| 18 | P4 | C | 正常（略差于19/20） | 正常 | 继电，减抖脚不动 |
| 19 | P4 | D | 正常 | 正常 | 继电，减抖脚不动 |
| 20 | P4 | E | 正常 | 正常 | 继电，减抖脚不动 |

---

## 三、21-33 组新增设计

### 新增设计目标

21-33 组不是重新大范围搜索，而是基于当前最可用的方向继续改：

- 基础组合：Group 19 的思路，即 P4 + Reward D
- 不再给 command=0 时额外加 `r_vel`
- 优先抑制静止时的上半身抖动、躯干晃动、动作输出跳变
- 另外补一个 P5 prior，用 `g1_low_walk.csv` 补 1m/s 以内低速行走数据，验证继电特性是否来自 prior 低速缺失

### 新增 reward 公式

公共定义：

```text
m_still = 1[||v_cmd|| <= 0.2]
r_vel = steering_target_velocity(vel_err_scale=0.5)
r_root_stop = exp(-2.0 * ||v_root_xy||^2)
r_joint_vel = exp(-0.02 * sum(|qdot|))
```

Reward F：躯干角速度惩罚

```text
r_root_ang = exp(-1.0 * ||omega_root_xy||^2)
r_task = (1-m_still) * r_vel
       + m_still * (r_root_stop * r_joint_vel * r_root_ang)
```

Reward G：上半身关节速度惩罚

```text
r_upper_joint_vel = exp(-0.05 * sum(|qdot_upper|))
r_task = (1-m_still) * r_vel
       + m_still * (r_root_stop * r_joint_vel * r_upper_joint_vel)
```

`qdot_upper` 目前选取 waist、shoulder、elbow、wrist 相关关节。

Reward H：输出平滑

```text
r_action_smooth = exp(-0.25 * ||a_t - a_{t-1}||^2)
r_task = (1-m_still) * r_vel
       + m_still * (r_root_stop * r_joint_vel * r_action_smooth)
```

这些 reward 仍然通过 `task_smp_product` 使用，所以总形式仍是：

```text
r_final = r_task * r_smp
```

### 新增 group 映射

| 组 | Prior | Reward | Command | 设计目的 |
|----|-------|--------|---------|---------|
| 21 | P4 | F | Exp6-G7 原 command | 在 19 基础上抑制躯干 roll/pitch 角速度 |
| 22 | P4 | G | Exp6-G7 原 command | 在 19 基础上抑制上半身关节速度 |
| 23 | P4 | H | Exp6-G7 原 command | 在 19 基础上抑制 action 抖动 |
| 24 | P4 | F | 速度小于 1.0 置 0 | 同 21，但加入低速死区 |
| 25 | P4 | G | 速度小于 1.0 置 0 | 同 22，但加入低速死区 |
| 26 | P4 | H | 速度小于 1.0 置 0 | 同 23，但加入低速死区 |
| 27 | P5 | F | Exp6-G7 原 command | 用低速 prior 替换 P4，验证低速数据影响 |
| 28 | P5 | G | Exp6-G7 原 command | 用低速 prior 替换 P4，验证低速数据影响 |
| 29 | P5 | H | Exp6-G7 原 command | 用低速 prior 替换 P4，验证低速数据影响 |
| 30 | P5 | F | 速度小于 1.0 置 0 | P5 + 躯干角速度惩罚 + 低速死区 |
| 31 | P5 | G | 速度小于 1.0 置 0 | P5 + 上半身关节速度惩罚 + 低速死区 |
| 32 | P5 | H | 速度小于 1.0 置 0 | P5 + action smooth + 低速死区 |
| 33 | P5 | D | Exp6-G7 原 command | P5 + group19 baseline，用来单独隔离低速 prior 的影响 |

说明：21-26 是 P4 上的 reward/command 改动；27-32 是把 21-26 的 prior 换成 P5；33 是额外保留的 P5 baseline 对照。

### P5 prior 的数据和生成

P5 数据组合：

```text
forward + stop_static + g1_low_walk
```

预期新增动作路径：

```text
datasets/csv/forward/g1_low_walk.csv
```

生成 P5 prior：

```bash
bash scripts/run_exp9_prepare_priors.sh prior5
```

输出：

```text
datasets/pretrain_ckpt/exp9_prior5_forward_stop_static_low_walk.pt
```

### 新增训练和 play 脚本

训练单组：

```bash
bash scripts/run_exp9_policy_groups1_33.sh <group> <gpu>
```

例子：

```bash
bash scripts/run_exp9_policy_groups1_33.sh 21 0
bash scripts/run_exp9_policy_groups1_33.sh 24 1
bash scripts/run_exp9_policy_groups1_33.sh 27 2
bash scripts/run_exp9_policy_groups1_33.sh 33 3
```

Play 单组：

```bash
bash scripts/play_exp9_groups1_33_wandb.sh --gpu 0 <group> <wandb_run_path>
```

例子：

```bash
bash scripts/play_exp9_groups1_33_wandb.sh --gpu 0 21 <wandb_run_path>
bash scripts/play_exp9_groups1_33_wandb.sh --gpu 1 --video-length 1500 30 <wandb_run_path>
```

---

## 四、按 Prior 分析

### P1（forward + walk_to_stop）
- Reward A/B：运动正常，但静止时抖，和实验七 baseline 没本质区别
- Reward C：**运动正常 + 减抖**，但脚不动
- Reward D/E：静止效果和 C 差不多，但运动能力被破坏（D 跑步不对称、E 完全崩）
- **结论**：P1 + C 是 P1 里最好的组合，但"脚不动"问题未解决

### P2（forward + walk_to_stop + stop_to_walk）— 最差 prior
- 只有 A 保持运动（但静止仍抖），B/C/D/E 全部崩
- **结论**：加入 stop_to_walk 数据严重破坏了 prior 质量，可能因为 stop_to_walk 动作本身质量差或和 forward 数据冲突

### P3（forward + walk_to_stop + stop_to_walk + stop_static）
- 目前只有 E 有结果（崩 + 减抖脚不动），和 P2-E 表现一致
- **结论**：stop_static 没有挽救 P2 的崩盘，待 A-D 测试确认

### P4（forward + stop_static）— 最好 prior
- A：运动正常，静止抖（和 P1-A 一样）
- B：完全不停（门控方式在 P4 上失效）
- C/D/E：**运动正常 + 减抖脚不动**
- **结论**：P4 是最好的 prior——只加 stop_static，不加切换动作，prior 质量最高。C/D/E 三种 reward 都能减抖且不破坏运动

### P5（forward + stop_static + g1_low_walk）
- 目的不是替代 P4，而是专门验证低速数据是否能缓解继电特性
- 如果 P5 + D baseline（Group 33）比 P4 + D（Group 19）更容易从 0 到 1m/s 起步，说明继电问题主要来自 prior 缺少低速动作
- 如果 P5 仍然继电，问题更可能来自 command 分布、reward 切换边界或 policy 学习过程

---

## 五、按 Reward 分析

### Reward A（简单关节速度惩罚）
- 所有 prior 下运动能力都保持（P1/P2/P4 正常）
- 但静止效果差：都只是"0速下抖"，没有本质改善
- **特点**：最保守，不伤运动，但静止效果不够

### Reward B（门控）
- P1：运动正常但抖；P2：崩；P4：完全不停
- **特点**：门控方式不稳定，在不同 prior 上表现差异极大，不可用

### Reward C（加权和切换）
- P1：运动正常 + 减抖；P2：崩；P4：运动正常 + 减抖
- **特点**：在好 prior（P1/P4）上能减抖，在差 prior（P2）上会崩。适中

### Reward D（乘积切换）
- P1：跑步不对称 + 不会走；P2：崩；P4：运动正常 + 减抖
- **特点**：乘积惩罚太强，在 P1 上已经伤运动，但在 P4 上表现好

### Reward E（乘积混合切换）
- P1：完全崩；P2/P3：崩；P4：运动正常 + 减抖
- **特点**：最复杂的 reward，在 P4 上和 C/D 效果一致，但在其他 prior 上都崩。只在最好的 prior 上才能用

### Reward F（D + 躯干角速度惩罚）
- 直接针对静止时躯干晃动
- 如果 F 有效，应当看到上半身摆动频率下降，但脚不应明显恢复乱挪
- 风险是过强的 root 姿态稳定会让起步更难，继电特性可能更明显

### Reward G（D + 上半身关节速度惩罚）
- 直接针对你观察到的上半身高频抖动
- 比惩罚全身关节更聚焦，理论上更不容易破坏腿部重新起步
- 风险是上肢/腰部被压得太死，可能影响 SMP style 或平衡调整

### Reward H（D + action smooth）
- 不直接规定身体姿态，而是抑制 policy 输出跳变
- 如果抖动来自策略输出高频抖，H 可能更有效
- 风险是动作响应变慢，低速到起步的切换可能更迟钝

---

## 六、到底可以用什么

### 目前最好的组合：Group 19/20（P4 + Reward D/E）

| 指标 | Group 19 (P4+D) | Group 20 (P4+E) | Group 18 (P4+C) |
|------|-----------------|-----------------|-----------------|
| 跑步 | 正常 | 正常 | 正常（略差） |
| 走路 | 正常 | 正常 | 正常 |
| 静止 | 继电，减抖脚不动 | 继电，减抖脚不动 | 继电，减抖脚不动 |

- **Prior 用 P4**（forward + stop_static），不要加切换动作
- **Reward 用 D 或 E**，C 也可以但运动略差
- 三者静止表现一致：抖动大幅减少，但上身抖、脚不动

### 新增阶段优先关注的组合

| 目的 | 优先看 |
|------|--------|
| 只看躯干角速度惩罚是否减少晃动 | Group 21 vs Group 19 |
| 只看上半身关节惩罚是否减少晃动 | Group 22 vs Group 19 |
| 只看 action smooth 是否减少高频输出 | Group 23 vs Group 19 |
| 看 command dead zone 是否缓解低速抖动 | Group 24-26 vs Group 21-23 |
| 看低速 prior 是否缓解继电 | Group 33 vs Group 19 |
| 看低速 prior + 新 reward 是否更好 | Group 27-32 vs Group 21-26 |

### 不行的方案
- **P2 prior**（加了 stop_to_walk）：几乎全崩，不要用
- **Reward B**（门控）：不稳定，不要用
- **P1 + D/E**：破坏运动能力，不要用

---

## 七、未解决的问题 & 基于什么去改进

### 问题 1：继电特性（缺少低速运动）
- 所有组都有继电特性，机器人速度切换不连续
- **改进方向**：
  - 调整 command 分布，增加低速 speed 的采样概率
  - 设置死区（speed 低于阈值直接归零）
  - 延长静止指令的时间（如 10 秒），让机器人多经历静止训练

### 问题 2：0 速下上身抖、脚不动
- C/D/E reward 能减抖，但代价是脚不动——静止 reward 太强，把所有运动都压死了
- **改进方向**：
  - 降低静止 reward 中关节速度惩罚的权重（如把 `r_joint_vel` 的系数从 0.4 降到 0.2）
  - 只惩罚脚部关节速度，不惩罚全身（区分上身和下身）
  - 对身体晃动做额外惩罚（针对上身抖）
  - 尝试只惩罚 root 速度不惩罚关节速度，看脚能不能自然着地

### 问题 3：P2 为什么崩
- 加入 stop_to_walk 后 prior 质量严重下降
- **待查**：stop_to_walk.csv 数据本身质量如何？是否和 forward 数据分布差异过大导致 prior 学混了？

### 问题 4：P3 待测试（11-14 组）
- P3 = P2 + stop_static，目前只有 E 崩了
- 需要测 A-D 确认 stop_static 能否挽救 P2 的崩盘
- 如果 P3-A 也不行，说明 stop_to_walk 本身就是毒药

### 问题 5：sim2sim 验证
- 目前最好效果（Group 19/20）还没有做 sim2sim
- 需要用实机部署指令训练后做 sim2sim 验证

---

## 八、下一步建议

1. **优先测 P3 的 11-14 组**，确认 stop_to_walk 是否是问题根源
2. **基于 Group 19/20 继续优化**：
   - 加死区 + 调整 command 分布
   - 降低关节速度惩罚权重 / 只惩罚脚部
   - 延长静止指令时间
3. **对 Group 19/20 做 sim2sim**，验证实际部署效果
4. 如果 P3-A 正常，说明 stop_static 能抵消 stop_to_walk 的负面影响，可以尝试 P3 + A
