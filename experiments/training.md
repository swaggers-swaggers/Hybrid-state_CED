# 检测头分布监督与分块训练

本实验目前关注每个有效位置上检测头的输出是否改善。常规流程为 `modules → confidence → evaluate`，不跑推理延迟或加速比测试。`calibrate` 是可选的独立分析，不参与检测头是否被评估的决策。

## 数据、来源与监督位置

使用已审计的 teacher Top-8 稀疏数据。第 `j` 行 logits 对应 teacher 在前缀 `prompt + y[:j]` 下对 `y[j]` 的预测。训练输入 `y[j-1]`，检测头监督第 j 行；该步若退出，下一步输入 teacher `y[j]`，完整 student 输出监督第 `j+1` 行。prompt、首个生成位置、padding 和被排除的回答不计损失，EOS 按数据 mask 计入。

默认载入 `results/training/20260923-top8-warm-8h/experiment/calibrated.pt` 的 M12、三个投影器、置信头，逐项确认参数与来源相等。它已处理有效区间 `[0,500000)`；当前默认从 500,000 继续。来源 checkpoint 的模型和数据身份必须匹配。相同实验语义的后续 checkpoint 还必须匹配配置；发生实验语义变化时只允许已知来源的显式迁移，并重新记录来源哈希，不能沿用其中的判定阈值。

## 损失与可信判定

设 teacher Top-8 ID 集合为 S。每个概率使用全词表归一化：`p_i = exp(logit_i/T - logsumexp_all_T)`，第九类概率为 `1 - sum_{i∈S} p_i`。student 同样对 S 和其余词表聚合；尾部使用剩余 logits 的 logsumexp，保留梯度。`T=1` 为默认，也支持数据中已保存统计的 `T=2`，蒸馏损失乘 `T²`。

- 检测头：所有有效位置的九类 KL 求均值。可信标签、置信网络输出和是否形成 KV 对，都不影响 M12 是否接收该位置梯度。
- KV 投影器：teacher 条件判定为 `Top8 overlap ≥ 6 OR Top1 match` 且有下一位置时退出。P16/P20/P24 仅用 h12 生成当前输入位置的 K/V。紧接下一步一定完整执行，哪怕该位置自身也符合可信规则。该步的完整输出九类 KL 是投影器唯一的数据监督，不用 KV MSE、目标状态替换或 GDN 偏差指标。
- 置信网络：独立 gate 集上的上述 OR 标签；BCE，无重置。每个 epoch 后按 dev BCE 选择，源参数也作为候选。它只学习预测规则标签，不能筛掉检测头的训练或评估样本。

M12 与投影器使用独立 AdamW，默认学习率分别为 `1e-4`、`1e-5`，weight decay `0.01`，梯度裁剪 1。两者分别按实际监督位置数、已完成的监督对数归一化梯度。没有完成对的更新周期不对投影器执行 step，也不施加 weight decay。置信网络学习率 `3e-4`，batch 1024，5 epoch。

## GPU 利用率的实现改动

1. 前 12 层冻结，在有序 teacher 前缀上分块执行，每块默认 32 个输入位置。一次因果序列计算得到整块 h12，避免把浅层所有线性运算拆成逐 token 的小矩阵乘法。
2. M12 与冻结词表对整块 h12 一起计算，全部位置的损失一起反向传播。最多保留当前块的全词表 student logits；数据侧仍只有稀疏 teacher Top-8。
3. 浅层与深层使用独立缓存，初始前缀一致。浅层可以先算完整块；深层只按位置读取该块对应的 h12，未处理的位置不能进入深层 KV。单查询、无 padding 的深层 attention 可以读取全部已积累 key；RoPE 使用该位置的绝对位置。
4. 深层 KV/GDN 的状态演进继续按 token 顺序执行。正常完整步若不承担投影监督，只推进深层状态，不算最终 Norm/词表 logits；投影后的完整下一步才计算输出蒸馏并反向。
5. 可信判定只用 Top-8/Top-1，移除重复计算判定 KL。teacher 九类概率每条回答校验一次；分支布尔向量每块传回 CPU 一次，训练损失累计留在 GPU，避免逐 token `.item()` 同步。
6. 使用 fused AdamW。完成成对监督后才允许改变投影参数；若退出位于分块最后，下一块第一步完成监督后才能清理图与更新。优化器可跨块累积，绝不在未完成监督对时 step。
7. 开发集、测试集和置信标签采集只计算前 12 层与检测头，不运行被置信拒绝后的完整模型。纯检测头指标不需要深层计算。

这些改动减少串行调度与小矩阵操作；实际训练吞吐/GPU 利用率的提升仍需后续训练的运行记录确认。本次不以推理开销测试证明训练优化。分块 BF16 与逐步 BF16 的运算顺序可能带来小数值差异，正确性校验包括损失差异和因果性检查。

## 每次实验报告什么

`modules` 训练前后在同一 dev 子集生成 `detector_before`、`detector_after`、`detector_delta`。最终 `evaluate` 在 test 集报告：

- Top-1 一致率、Top-8 平均重合数、重合数 0–8 的直方图、Top-8 recall（重合数/8）。
- 九类 KL 与尾部质量绝对误差，另报告仅在 teacher Top-8 内重新归一化后的条件 KL，帮助区分候选内部概率与尾部质量的问题。
- teacher 生成 token 的 NLL 及 teacher 自身该 token 的 NLL；输出明确来自检测头。
- OR 条件合格率及置信网络 BCE。合格率不改变前述指标的分母。

九类 KL、条件 KL、NLL 越低越好；一致率和平均重合数越高越好。所有位置均进入评估，结果显式标注 `confidence_filter_applied=false`。这些是 teacher 前缀下的分布能力指标，不等同于自由生成任务准确率。

## 执行与限时

默认 `--plan` 不训练。例子（在仓库根目录，输出目录需不存在）：

```bash
/home/liu/miniconda3/envs/ai/bin/python scripts/train_top8_distill.py --run --stage all \
  --start 500000 --tokens 500000 --evaluation-responses 64 \
  --output results/training/<本次运行目录>
```

`--checkpoint` 可继续已完成的当前语义 checkpoint；主训练起点必须等于源 checkpoint 区间结束位置。`--stage evaluate` 可以直接读取 modules 或 confidence checkpoint，无需先校准。`--smoke --run` 仍是真实训练，`--verify` 则不更新权重。

需要延续八小时以内预算时使用 `scripts/run_top8_bounded.py --output <全新目录>`：主模块最多 6 小时，整个 worker 7.5 小时软件截止、外层 7 小时 50 分硬截止；数据量按开始阶段吞吐保守确定。整段训练结束后再做开发评估，不在途中查看指标。分块执行按已完成位置记录实际区间，达到预算后完成已经开始的退出—完整步对再保存。

主模型不复制，辅助 checkpoint、日志与报告每个运行目录上限 2 GB、磁盘至少保留 20 GB。实验细节按日期写在 `experiments/records/`，README 保留目标、架构、监督和入口。
