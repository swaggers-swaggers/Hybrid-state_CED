# K/V 缓存注入与续写小规模实验

结论：**HOLD_REPAIR_KV**

固定已有 checkpoint；256/2048 长度各 16 个测试窗口；每个窗口 128 步相同 token 续写。保留 teacher GDN 状态。
Step 1 为最后一个 prompt token 经过缓存路径后预测首个续写 token。测试窗口互不重叠，但来自 Phase 1 已用过的 test split；并非新的独立语料。
门槛在运行前固定，是本轮工程筛选标准，不代表普适质量阈值。未测试自由生成、GDN 重建或实际加速。

| Context | Method | Mean KL | ΔNLL (nats/token) | Top-1 agreement | KL@1 | KL@32 | KL@128 |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 256 | oracle_reencoded | 0.000000 | 0.000000 | 100.00% | 0.000000 | 0.000000 | 0.000000 |
| 256 | predicted_15 | 0.016095 | 0.017707 | 94.73% | 0.009906 | 0.008992 | 0.011136 |
| 256 | predicted_19 | 0.022300 | 0.043816 | 94.48% | 0.025889 | 0.023048 | 0.013270 |
| 256 | predicted_23 | 0.028525 | 0.020670 | 93.80% | 0.109814 | 0.061398 | 0.015458 |
| 256 | predicted_all | 0.111516 | 0.153414 | 88.13% | 0.084881 | 0.083159 | 0.054387 |
| 256 | original_all | 0.089650 | 0.101170 | 89.26% | 0.053940 | 0.044316 | 0.032569 |
| 256 | zero_all | 0.269706 | 0.329056 | 82.86% | 0.181192 | 0.146315 | 0.064006 |
| 2048 | oracle_reencoded | 0.000000 | 0.000000 | 100.00% | 0.000000 | 0.000000 | 0.000000 |
| 2048 | predicted_15 | 0.043124 | 0.047463 | 90.28% | 0.033973 | 0.014028 | 0.021208 |
| 2048 | predicted_19 | 0.050620 | 0.056867 | 92.72% | 0.061556 | 0.046849 | 0.014741 |
| 2048 | predicted_23 | 0.063457 | 0.061779 | 90.38% | 0.048592 | 0.026293 | 0.025268 |
| 2048 | predicted_all | 0.293027 | 0.296886 | 80.76% | 0.258678 | 0.124709 | 0.087864 |
| 2048 | original_all | 0.185859 | 0.221720 | 82.57% | 0.190557 | 0.113512 | 0.054879 |
| 2048 | zero_all | 0.701910 | 0.746799 | 70.56% | 0.585086 | 0.317807 | 0.164554 |

长度 256：HOLD_REPAIR_KV；检查：`{"beats_both_baselines": false, "mean_kl_acceptable": false, "boundary_kl_acceptable": true, "nll_increase_acceptable": false, "no_large_late_growth": true}`

基线比较（以窗口为单位配对 bootstrap，不能代表多随机种子训练置信度）：
```json
{
  "zero_all": {
    "relative_improvement": 0.5865272145602107,
    "paired_sequence_bootstrap_ci95": [
      0.5415664995490415,
      0.6273100507807466
    ]
  },
  "original_all": {
    "relative_improvement": -0.24390446199442928,
    "paired_sequence_bootstrap_ci95": [
      -0.5231293925003757,
      -0.06283159489285417
    ]
  }
}
```


长度 2048：HOLD_REPAIR_KV；检查：`{"beats_both_baselines": false, "mean_kl_acceptable": false, "boundary_kl_acceptable": false, "nll_increase_acceptable": false, "no_large_late_growth": true}`

基线比较（以窗口为单位配对 bootstrap，不能代表多随机种子训练置信度）：
```json
{
  "zero_all": {
    "relative_improvement": 0.5825291165436062,
    "paired_sequence_bootstrap_ci95": [
      0.5543371239670702,
      0.6036591686472933
    ]
  },
  "original_all": {
    "relative_improvement": -0.5766097290328056,
    "paired_sequence_bootstrap_ci95": [
      -0.7945420160207711,
      -0.40912790432448193
    ]
  }
}
```


所有 batch 均验证 teacher K/V 经 KNorm/RoPE 重建逐位相等，oracle 128 步 logits 与 teacher 逐位相等。
运行耗时：120.4 秒。

决策规则：两种长度均通过，才支持进入单层 GDN pilot；否则先修复 K/V 或收缩替换范围，不据此否定所有 CED 方案。
