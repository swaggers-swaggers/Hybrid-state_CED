# K/V 续写实验口径与目标捕获审计

本轮仅判断现有 Phase 1 checkpoint 是否足以支持进一步投入 GDN 重建，不训练新模型，不据本轮结果调参。冻结配置见 `configs/kv_continuation_pilot.json`，入口为 `scripts/run_kv_continuation_pilot.py`。

## 固定实验口径

- WikiText-103 test token 流中随机选取不重叠窗口，256 与 2048 长度各 16 个；后续 128 个 token 来自同一连续 token 流。该 test split 已用于 Phase 1，不能称为新的独立确认集。
- 每种长度共 2048 个预测位置；置信区间以 16 个窗口为重采样单位，而非将全部 token 当作独立样本。窗口仍可能来自同一文章，区间应作探索性证据。
- teacher 先处理前 N−1 个 prompt token；复制其全部异构缓存，仅替换指定层的 K/V。第 1 步消耗最后一个 prompt token，预测首个 continuation token；每条分支随后消耗相同的真实文本 token。
- 对比 teacher、teacher 真实 K/V 重新编码、分别替换 layer 15/19/23、三层同时替换、三层 original projection、三层 zero K/V。zero K/V 保留历史槽位和位置，是零缓存对照，不等于删除历史槽位的 state-dropping 实现。
- original projection 沿用 Phase 1 的定义：直接对 H12 使用目标层 k_proj/v_proj；不将它声称为所有可能归一化基线中最强的方案。
- K 使用目标层 KNorm 与模型 RoPE 转换为实际缓存形式；V 按原注意力头布局写入。所有 GDN 状态保留 teacher 值。这是隔离 K/V 误差的诊断，不是跳过上层 prefill 的端到端实现。
- 所有 batch 必须通过实际 teacher K/V 重建逐位一致、独立缓存存储校验；oracle 的 128 步 logits 必须逐位等于 teacher。失败即停止，不给方法判定。

## 运行前固定的工程门槛

两种长度下三层同时替换均满足以下条件，才标记 GO_TO_GDN_PILOT：

1. 相对 original/zero 两条基线的窗口平均 KL 均改善至少 10%，配对 bootstrap 95% 区间下界大于 0。
2. 全 128 步平均 KL ≤ 0.1 nats；首 token 平均 KL ≤ 0.1 nats。
3. 相对 teacher 的平均 NLL 增加 ≤ 0.1 nats/token。
4. 第 97–128 步平均 KL ≤ 第 1–8 步平均 KL 的 2 倍 + 0.01。

否则标记 HOLD_REPAIR_KV，表示暂停扩展，先修复 K/V 路径或收缩替换范围，并非否定所有 CED 路线。阈值是小规模工程筛选标准，不能解释为普适性能保证。逐步 KL 完整保存，末段增长检查不能保证任意更长续写均稳定。

## Phase 1 目标捕获存在口径错误

本轮核对已安装 Transformers 的实际 decoder forward 后确认：

1. `phase1/teacher.py` 的 `_target_hook` 挂在整个 decoder layer 的 pre-hook，取得的是 input_layernorm **之前**的 residual hidden。
2. `teacher_raw_kv` 直接将这一 hidden 输入 k_proj/v_proj；`normalized_qk` 也直接用它生成 Query，没有应用该层 input_layernorm。
3. 实际模型先执行该层 input_layernorm，再把结果传给 self_attn；真实 K/V 和 Query 均来自这个归一化之后的 hidden。

因此现有 Phase 1 数字证明的是这个代理目标上的可预测性，不能直接证明真实 attention cache 的可恢复性。仅在 K 上施加 KNorm 并不能替代缺失的输入归一化，尤其 V 没有对应的 KNorm。

本轮为保持现有 checkpoint 的可审计性，不修改或覆盖 Phase 1 的代码、权重和原始结果；实验使用实际 self_attn 输入构建 oracle，并把原 checkpoint 预测结果注入真实推理路径。旧目标错误的具体损害需要结合本轮结果判断，不能仅凭代码审计推断其全部数值影响。

若本轮未通过，下一轮应把目标捕获改到 self_attn 输入（支持 keyword arguments），加入预测前真实 K/V 的一致性断言，然后重新训练与评估。旧 checkpoint 不能仅通过修改评估脚本就视为已修复；新旧产物应分目录保存。
