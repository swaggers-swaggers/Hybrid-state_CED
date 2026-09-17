# K/V 缓存注入与续写小规模实验

结论：**HOLD_REPAIR_KV**

固定已有 checkpoint；256/2048 长度各 16 个测试窗口；每个窗口 128 步相同 token 续写。保留 teacher GDN 状态。
Step 1 为最后一个 prompt token 经过缓存路径后预测首个续写 token。测试窗口互不重叠，但来自 Phase 1 已用过的 test split；并非新的独立语料。
门槛在运行前固定，是本轮工程筛选标准，不代表普适质量阈值。未测试自由生成、GDN 重建或实际加速。

| Context | Method | Mean KL | ΔNLL (nats/token) | Top-1 agreement | KL@1 | KL@32 | KL@128 |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 256 | oracle_reencoded | 0.000000 | 0.000000 | 100.00% | 0.000000 | 0.000000 | 0.000000 |
| 256 | predicted_15 | 0.045019 | 0.039207 | 88.38% | 0.034531 | 0.050828 | 0.021136 |
| 256 | predicted_19 | 0.049124 | 0.038894 | 89.70% | 0.040325 | 0.049019 | 0.018354 |
| 256 | predicted_23 | 0.067251 | 0.078396 | 88.67% | 0.076430 | 0.057040 | 0.044431 |
| 256 | predicted_all | 0.267799 | 0.246444 | 78.37% | 0.221915 | 0.161156 | 0.075859 |
| 256 | original_all | 0.242187 | 0.223898 | 80.42% | 0.238262 | 0.128007 | 0.063079 |
| 256 | zero_all | 0.311142 | 0.286552 | 78.42% | 0.523914 | 0.204786 | 0.083363 |
| 2048 | oracle_reencoded | 0.000000 | 0.000000 | 100.00% | 0.000000 | 0.000000 | 0.000000 |
| 2048 | predicted_15 | 0.075241 | 0.085114 | 86.23% | 0.063390 | 0.269359 | 0.066125 |
| 2048 | predicted_19 | 0.082337 | 0.078931 | 89.26% | 0.122985 | 0.120542 | 0.046520 |
| 2048 | predicted_23 | 0.099133 | 0.131796 | 84.86% | 0.143933 | 0.183319 | 0.059989 |
| 2048 | predicted_all | 0.461666 | 0.472021 | 71.73% | 0.977273 | 0.604132 | 0.221876 |
| 2048 | original_all | 0.465333 | 0.482816 | 73.05% | 1.175737 | 0.619581 | 0.153721 |
| 2048 | zero_all | 0.676995 | 0.672200 | 67.38% | 2.423485 | 0.873760 | 0.200457 |

长度 256：HOLD_REPAIR_KV；检查：`{"beats_both_baselines": false, "mean_kl_acceptable": false, "boundary_kl_acceptable": false, "nll_increase_acceptable": false, "no_large_late_growth": true}`

基线比较（以窗口为单位配对 bootstrap，不能代表多随机种子训练置信度）：
```json
{
  "zero_all": {
    "relative_improvement": 0.13930229181387999,
    "paired_sequence_bootstrap_ci95": [
      0.03884412402906775,
      0.2262681151808448
    ]
  },
  "original_all": {
    "relative_improvement": -0.1057515194680565,
    "paired_sequence_bootstrap_ci95": [
      -0.20127397646017026,
      -0.02963824088007717
    ]
  }
}
```



长度 2048：HOLD_REPAIR_KV；检查：`{"beats_both_baselines": false, "mean_kl_acceptable": false, "boundary_kl_acceptable": false, "nll_increase_acceptable": false, "no_large_late_growth": true}`

基线比较（以窗口为单位配对 bootstrap，不能代表多随机种子训练置信度）：
```json
{
  "zero_all": {
    "relative_improvement": 0.3180655223147083,
    "paired_sequence_bootstrap_ci95": [
      0.26162213049855654,
      0.36162515149848395
    ]
  },
  "original_all": {
    "relative_improvement": 0.00787932006444314,
    "paired_sequence_bootstrap_ci95": [
      -0.038953713018942515,
      0.04457948787977794
    ]
  }
}
```


所有 batch 均验证 teacher K/V 经 KNorm/RoPE 重建逐位相等，oracle 128 步 logits 与 teacher 逐位相等。
运行耗时：119.4 秒。

决策规则：两种长度均通过，才支持进入单层 GDN pilot；否则先修复 K/V 或收缩替换范围，不据此否定所有 CED 方案。

## 本轮解读与后续决策

三层同时替换在 256 长度下的平均 KL 比 original projection 高 10.58%；在 2048 长度下只降低 0.79%，配对区间跨过零。虽然两种长度均优于 zero K/V，但尚不具备稳定优于原始投影的证据。预测三层的平均 NLL 分别增加 0.2464 和 0.4720 nats/token，首 token KL 分别为 0.2219 和 0.9773，均未通过运行前固定门槛。

单层替换的平均 KL 较小（256 下 0.0450–0.0673；2048 下 0.0752–0.0991），三层联合误差明显更大。本轮末段平均 KL 未出现门槛所定义的大幅增长，主要失败是替换与边界误差以及未优于原始投影；不能笼统称为递归误差爆炸。

代码审计确认 Phase 1 的目标 hidden 捕获在 input_layernorm 之前，与真实 attention 输入不同。因而原 Phase 1 GO 应降级为代理目标结果，不能继续据此推进 GDN。该错误是已确认的实现问题，但本轮没有修复后对照，不能把全部续写损失都归因于它。具体证据见 [口径审计](protocol_and_audit.md)。

建议：暂停 GDN 扩展；修复目标捕获并重新训练后再做同类续写验证，使用新的确认数据以降低反复使用 test split 的偏差。不直接否定整个研究方向。

验证：新增 4 项门槛逻辑测试通过；全套 19 项测试运行成功，其中 3 项独立 GPU 测试按默认配置跳过。本轮实验本身在真实 GPU 上完成，包含全部窗口的真实 K/V 重建和 128 步 oracle 精确一致性校验。
