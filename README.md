# CED：Qwen3.5-0.8B 检测头与 KV 投影

当前目标是提高第 12 层检测头与完整原模型的**输出分布一致性**。训练和评估都在每个有效位置计算检测头输出，不因置信判定拒绝而跳过监督，也不以完整模型回退结果替代检测头指标。后续常规实验不测推理开销。

## 模型与数据

- 主模型：本地 `models/Qwen3.5-0.8B-Base/`；主干、原词表矩阵和 Norm 冻结。
- 一个检测头位于 block 12：`h12 → 可训练矩阵 M12 → 冻结词表 → logits`。
- 一个置信网络 `C(h12)`；三个独立投影器 `P16/P20/P24(h12)` 分别补写目标 full attention 层的 KV。早退时跳过的深层 GDN 保持状态。
- 数据：`data/teacher_top8_20260920/dataset/`。teacher 主训练集有 5,000,119 个有效位置，稀疏保存 Top-8 ID/logits、归一化统计和生成 token；不保存全词表 logits。
- 默认继承最近完成的全部辅助模块权重：`results/training/20260923-top8-warm-8h/experiment/calibrated.pt`。已消费区间 `[0,500000)`，接续默认从位置 500,000 开始，置信头也不重新初始化。

## 三类模块如何训练

把 teacher 的 8 个候选各作为一类，再把其余词表概率合并成一类，得到九类分布 `p⁹`；student 在同一组 token ID 上计算 `q⁹`。默认温度 `T=1`，KL 使用全词表归一化后的概率。

| 模块 | 监督与损失 | 更新范围 |
|---|---|---|
| 检测头 M12 | **全部有效位置**：`L_head = mean KL(p⁹_teacher(t) ∥ q⁹_head(t))` | 只更新 M12，可信和不可信位置都反向传播 |
| 三个 KV 投影器 | 位置 t 符合可信条件时退出并补 KV；紧接 t+1 输入 teacher token、强制完整执行：`L_KV = mean KL(p⁹_teacher(t+1) ∥ q⁹_full_with_projected_cache(t+1))` | 梯度经下一步完整输出传回 P16/P20/P24，不用 KV 数值对齐损失 |
| 置信网络 C | 标签 `y = [Top-8 重合数 ≥ 6] OR [Top-1 相同]`；`L_conf = BCEWithLogits(C(h12), y)` | 在独立 gate 划分继续训练，按 dev BCE 选权重，初始权重也作为候选 |

可信规则只有 **“Top-8 至少重合 6 个，或 Top-1 相同”**；KL 保留为损失和评估指标，不再作为可信门槛。训练使用 teacher token，prompt、padding、首个生成位置不计损失。末位置不发起缺少下一步监督的退出；每对结束后截断反向图并保留缓存值。

## 评估与执行

主要指标：检测头 Top-1 一致率、Top-8 平均重合数/重合分布、Top-8+其余词表 KL、Top-8 内部条件 KL、teacher 生成 token 的 NLL。训练前后在同一 dev 集比较，最终在 test 集直接评估检测头。置信分数不筛选这些位置；校准仅是可选分析，不能阻止检测头评估。

冻结的前 12 层与 M12/词表按默认 32 个位置分块计算；深层缓存按位置推进。无后续输出监督的完整步只更新深层状态，不做最终大词表读出。详见 [训练实现与指标说明](experiments/training.md)。

使用已有 `ai` 环境。默认只显示计划：

```bash
cd /home/liu/CED
/home/liu/miniconda3/envs/ai/bin/python scripts/train_top8_distill.py --plan
```

显式训练入口为 `scripts/train_top8_distill.py --run`；配置见 [qwen35_08b_top8_distill.json](configs/qwen35_08b_top8_distill.json)。每次使用新的输出目录；模型参数继承，优化器重建。磁盘产物每个运行目录上限 2 GB，至少保留 20 GB 可用空间。

## 文件入口

- [训练循环](ced_distill/training.py)、[损失与可信规则](ced_distill/losses.py)、[检测头评估](ced_distill/evaluation.py)。
- [实验记录目录](experiments/README.md)：具体配置、执行结果及实现校验，不在本 README 展开。
- `results/` 保存原始 JSON、日志和 checkpoint；`experiments/` 保存说明与解读，不复制模型或训练数据。
