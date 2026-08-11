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

## NewDev Exp13 G4-G6

三组均从零训练 10000 iterations，完全保留原 Exp13 G4-G6 的 policy 参数，只替换
为全镜像 prior。训练 checkpoint 使用 `SmpOnPolicyRunner` 保存：

- actor、critic、optimizer 和 iteration；
- mjlab environment state；
- `infos["smp_normalizer"]`；
- checkpoint lineage identity。

```bash
bash scripts/run_new_dev_exp13_prepare_prior.sh --gpu <gpu>
bash scripts/run_new_dev_exp13_groups4_6.sh <4|5|6> <gpu>
```

## NewDev Exp14 G1-G9

所有组都严格恢复对应 NewDev Exp13 checkpoint，并继续使用同一个全镜像 prior。
缺少 normalizer、normalizer 状态非法或 parent identity 不匹配时，训练会在恢复
actor/optimizer 前直接拒绝。

| Group | Parent | Added reward | Iterations |
|---:|---|---|---:|
| 1-3 | NewDev Exp13 G4/G5/G6 | none, continuation control | 100 |
| 4-6 | NewDev Exp13 G4/G5/G6 | moving debounced support-foot tilt, weight `-0.1` | 100 |
| 7-9 | NewDev Exp13 G4/G5/G6 | static left/right foot tilt, each weight `-0.1` | 100 |

```bash
bash scripts/run_new_dev_exp14_groups1_9.sh \
  <group> <gpu> <new_dev_exp13_wandb_path> [checkpoint]
```

```bash
bash scripts/play_new_dev_exp14_groups1_9_wandb.sh \
  --gpu <gpu> <group> <wandb_run_path>
```

Do not use the old `exp13` or `exp14` launcher entries for this chain. They refer to the
historical tasks and do not provide the NewDev runner/prior contract.
