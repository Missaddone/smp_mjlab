# new_dev: 可恢复 SMP 微调与 Exp16 设计

## 目标

在独立的 `new_dev` 分支中修复 SMP reward 归一化状态无法续训的问题，并建立两条互不混淆的训练链：重新生成可严格续训的 Exp13 G4--G6/Exp14 平脚微调链，以及使用 `motebu + stop_static` 及镜像数据训练新先验的 Exp16 链。

## 范围与不变量

- 不修改 `reconstruct`；所有实现位于 `new_dev`。
- 已有历史 RL checkpoint 不含 DiffNormalizer 状态，不得作为“严格连续”的 Exp14 父 checkpoint。
- Exp13 G4--G6 需从零重训；其 prior 使用原 Exp10 的 `loco + stop_static`
  数据源，但每个动作（含 `stop_static`）都必须成对镜像，输出
  `new_dev_exp13_loco_stop_static_mirrored.pt`。之后的 Exp14 只接受这些新 checkpoint。
- 所有新 prior 的原始数据恰为 `motebu.csv`、`motebu_mirror.csv`、`stop_static.csv`、`stop_static_mirror.csv`；包含静止数据的镜像，禁止漏镜像。
- Exp16 policy 从零训练，绝不 resume Exp13 actor；只比较 Exp13 G4/G5/G6 的三种移动 reward mix。
- 本次只交付可验证的实现与 GPU 启动脚本，不自动占用 GPU 执行长训练。

## 方案选择

采用严格状态连续方案，而不采用“旧 pt 缺状态时新建 normalizer 后继续”的兼容方案。后者可用于一般回放，但会把未知的 reward 标尺变化混入实验。new_dev 的精确恢复脚本会显式拒绝缺少 SMP normalizer state 的 checkpoint。

不尝试保存 MuJoCo 状态、GSI pool、motion buffer 或 RNG，因此本方案保证长期 reward 标尺连续，不宣称 bitwise 可复现。

## 架构

`DiffNormalizer` 提供受校验的 `state_dict/load_state_dict`。项目内
`SmpOnPolicyRunner(MjlabOnPolicyRunner)` 将该 state 保存在 checkpoint 的顶层
`infos['smp_normalizer']`，避免与上游 runner 覆盖的 `infos['env_state']` 冲突。
环境 startup 已在 runner load 前创建 `_smp_normalizer`，因此 runner load 可以安全恢复它。

严格模式仅用于 NewDev Exp14：从零训练的 NewDev Exp13 使用可保存 normalizer
状态的 runner；Exp14 才在恢复前拒绝旧 checkpoint。旧 checkpoint 在 strict resume
中抛出清晰错误；非严格使用则保持新建 state 并给出 warning。

## 实验链

### 可恢复 Exp13/Exp14

1. 先从 Exp10 的 `loco + stop_static` 源数据生成完整镜像的新 prior，再以原 Exp13
   G4、G5、G6 的参数分别从零训练 10,000 iteration，保存 new_dev checkpoint。
2. Exp14 对每个 parent 分开运行三种 100-iteration 低强度微调：无额外 reward 对照、移动支撑脚 tilt、静止双脚 tilt。微调使用较低 PPO 学习率、较小 KL 目标和较少 update epoch；三个因素不叠加。
3. 仅在上述结果选出 parent/单项后，才建立“移动+静止”组合组；本次不预先混合。

### Exp16 新先验

1. prepare 脚本建立隔离 staging 目录，只复制两份源 CSV，调用既有镜像工具生成两个镜像，断言总数为 4。
2. CSV-to-NPZ 与 DDPM 预训练复用 Exp10 的参数和工具，输出 `datasets/pretrain_ckpt/exp16_motebu_stop_static.pt`。
3. Exp16 三个 policy 均从零训练，分别采用 Exp13 G4/G5/G6 的移动 reward mix；prior 替换为 Exp16 prior，command 为 x/y `[-1.5,1.5]`、yaw `[-2,2]`。

## 验证

- normalizer round-trip、非法状态、runner 保存/恢复与旧 checkpoint strict-reject 都有单测。
- Exp16 prepare 脚本通过可替换命令的 shell 测试，验证严格四文件选择与镜像路径。
- 配置 builder 测试断言 Exp13/Exp14 strict runner、Exp16 prior 和 command/reward 参数。
- 对所有新 shell 脚本执行 `bash -n`；对 Python 执行 unittest、编译与 ruff。
- GPU 训练命令只输出，不执行。
