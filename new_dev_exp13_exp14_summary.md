# NewDev Exp13/Exp14: Resumable SMP Fine-Tuning

## Goal

在 `new_dev` 分支重新训练 Exp13 G4-G6，使 PPO checkpoint 同时保存
`DiffNormalizer` 状态，再从这些新 checkpoint 严格微调平脚奖励。历史 checkpoint
没有该状态，不能作为本链路的严格续训父模型。

## Prior

NewDev Exp13 prior 使用：

- `datasets/csv/loco/*.csv`
- `datasets/csv/forward/stop_static.csv`
- 每个原始动作的唯一镜像，包括 `stop_static_mirror.csv`

输出：

```text
datasets/pretrain_ckpt/new_dev_exp13_loco_stop_static_mirrored.pt
```

预训练显式复用 README 规定的全量 LAFAN normalization stats：

```text
datasets/norm_stats.npz
```

## 快捷运行入口

本分支统一使用原有 launcher：

```text
scripts/experiment_launcher.py
```

launcher 在当前终端前台运行，不创建 tmux、不后台启动。它会自动维护根目录下的
`wandb_run_registry.csv`，不需要手动查找或填写 W&B path。

先确认当前代码为 `new_dev`：

```bash
git branch --show-current
```

查看当前 NewDev Exp13/Exp14 对应的注册表内容：

```bash
grep -E '^exp(13|14),' wandb_run_registry.csv
```

## NewDev Exp13 G4-G6

三组均从零训练 10000 iterations，完全保留原 Exp13 G4-G6 的 policy 参数，只替换
为全镜像 prior。训练 checkpoint 使用 `SmpOnPolicyRunner` 保存：

- actor、critic、optimizer 和 iteration；
- mjlab environment state；
- `infos["smp_normalizer"]`；
- checkpoint lineage identity。

### 1. 准备 prior

```bash
bash scripts/run_new_dev_exp13_prepare_prior.sh --gpu 0
```

### 2. 训练三个父模型

每条命令都在当前终端前台训练。需要并行时，在三个终端中分别运行并调整 GPU：

```bash
uv run scripts/experiment_launcher.py train exp13 4 --gpu 0
```

```bash
uv run scripts/experiment_launcher.py train exp13 5 --gpu 1
```

```bash
uv run scripts/experiment_launcher.py train exp13 6 --gpu 2
```

启动时会生成新的 W&B run id，并分别覆盖：

- `exp13,4`
- `exp13,5`
- `exp13,6`

因此后续 Exp14 会自动使用本次 NewDev 重新训练的父模型，不会继续指向旧 Exp13。

Exp13 play：

```bash
uv run scripts/experiment_launcher.py play exp13 4 --gpu 0
```

将组号 `4` 替换为 `4..6` 即可。

## NewDev Exp14 G1-G9

所有组都严格恢复对应 NewDev Exp13 checkpoint，并继续使用同一个全镜像 prior。
缺少 normalizer、normalizer 状态非法或 parent identity 不匹配时，训练会在恢复
actor/optimizer 前直接拒绝。

| Group | Parent | Added reward | Iterations |
|---:|---|---|---:|
| 1-3 | NewDev Exp13 G4/G5/G6 | none, continuation control | 100 |
| 4-6 | NewDev Exp13 G4/G5/G6 | moving debounced support-foot tilt, weight `-0.1` | 100 |
| 7-9 | NewDev Exp13 G4/G5/G6 | static left/right foot tilt, each weight `-0.1` | 100 |

### 3. 训练 Exp14 G1-G9

launcher 自动从表格解析父模型：

- G1/G4/G7 读取 `exp13,4`；
- G2/G5/G8 读取 `exp13,5`；
- G3/G6/G9 读取 `exp13,6`。

不需要提供 `--source-wandb-path`。下面命令使用 GPU 0 作为示例，可自行修改：

```bash
uv run scripts/experiment_launcher.py train exp14 1 --gpu 0
uv run scripts/experiment_launcher.py train exp14 2 --gpu 0
uv run scripts/experiment_launcher.py train exp14 3 --gpu 0
uv run scripts/experiment_launcher.py train exp14 4 --gpu 0
uv run scripts/experiment_launcher.py train exp14 5 --gpu 0
uv run scripts/experiment_launcher.py train exp14 6 --gpu 0
uv run scripts/experiment_launcher.py train exp14 7 --gpu 0
uv run scripts/experiment_launcher.py train exp14 8 --gpu 0
uv run scripts/experiment_launcher.py train exp14 9 --gpu 0
```

每次启动会把对应的新 run id 覆盖到 `exp14,1..9`。如果父组的 Exp13 单元格为空，
launcher 会在启动训练前报错，不会要求手动输入 W&B path。

### 4. Play

play 同样自动读取 `exp14,<group>` 的 run id：

```bash
uv run scripts/experiment_launcher.py play exp14 1 --gpu 0
```

常用可选参数：

```bash
uv run scripts/experiment_launcher.py play exp14 1 \
  --gpu 0 \
  --num-envs 1 \
  --video-length 1500 \
  --viewer viser
```

将命令中的组号 `1` 替换为 `1..9` 即可播放对应组。

## 注册表覆盖说明

`new_dev` 分支有意复用并覆盖 `wandb_run_registry.csv` 中以下历史单元格：

- NewDev Exp13：`exp13,4..6`；
- NewDev Exp14：`exp14,1..9`。

`new_dev` 上的 `scripts/experiment_launcher.py` 已直接替换为 NewDev Exp13/Exp14
映射，不再保留历史任务入口。需要运行旧 Exp13/Exp14 时切换回原分支。
