# Phase 1 目标捕获修复

旧实现把 decoder layer 的归一化前 hidden 直接传给 K/V 投影，也用这个错误的 hidden 构建评估 Query。实际模型在 self-attention 前还会执行目标层 input_layernorm，因此旧训练目标与推理缓存不一致。

修复后的目标来自 self_attn 的真实输入，支持 keyword arguments。训练目标和 teacher Query 现在都经过模型原有的输入归一化；H4/H8/H12 source 仍保持原来的 residual 输出定义。Original Projection 对 H12 使用目标层 input_layernorm 后再投影，作为相同推理口径的基线。

每次训练及每个 validation/test 评估实例的首批数据都会启用真实缓存，检查捕获目标经 KNorm/RoPE/头布局转换后的 K/V 是否逐位一致。任何不一致都抛出错误并停止。完整训练只在首批进行该检查，避免为每批重复构建缓存。

新版 checkpoint 使用 schema version 2，并保存 target_semantics。评估拒绝旧口径权重，不能通过仅修改评估代码就把旧权重当作新训练结果。训练验证证据保存在 checkpoint 的 settings.target_cache_verification；评估验证证据保存在每个 split 的 target_cache_verification。

## 复验范围

- 保持原模型、训练数据、随机种子、优化器、学习率、batch size、低秩候选与训练规模；每层训练 4,999,936 tokens。
- 训练层 15/19/23；方法只按 validation 选择，test 不用于训练和方法选择。
- 新版权重写入 checkpoints/phase1_corrected，结果写入 results/phase1_corrected。旧版目录保留。
- 续写 pilot 沿用先前固定的质量、基线优势和误差增长门槛；256/2048 长度各 16 个窗口，128 步相同 token 续写。
- 续写窗口与首次 pilot 不重叠，但仍取自既有 test token 流；不能称为全新独立确认集。
- 新旧 pilot 的窗口、预测权重及原始投影基线均不同，不能把数值差异完全归因于单一修复。应以新版自身的配对比较和固定门槛判断是否继续。

## 复现

已有数据和首次 pilot 结果时，运行 `conda run -n ai python scripts/run_phase1_corrected.py`。标准入口 `scripts/run_phase1.sh` 会先准备数据，再进入同一流程。

流程依次执行数据哈希校验、包含真实模型的测试、三层完整训练/评估、跨层汇总、真实续写复验与最终测试。模型 NO-GO 是科学结果，仍完成其余层；执行错误、一致性断言失败或测试失败会停止。各阶段日志单独保存，结束后统一检查。
