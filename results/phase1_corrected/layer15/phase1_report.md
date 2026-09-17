# Qwen3.5-0.8B Phase 1 Attention KV Recoverability

**Gate 1：GO**

验证集选择 `low_rank_256`，其在独立测试集上同时改善 K/V reconstruction 与 teacher-query attention output。

## 实验设置

- Teacher：`/home/liu/CED/models/Qwen3.5-0.8B-Base`，revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`。
- Source：H4/H8/H12（decoder layer 3/7/11 输出）；Target：layer 15 raw pre-RoPE K/V。
- 训练：4,999,936 tokens，1,221 steps，455.4 秒。
- 序列长度：256；held-out validation/test 各 1,024 sequences。
- Functional metric：固定真实 teacher layer-15 Query，对 teacher/predicted K/V 应用原 KNorm 与 RoPE 后计算 causal attention output。
- Target 为 input_layernorm 之后的 self_attn 输入；首批目标经 KNorm/RoPE 后必须与真实缓存逐位相等。
- Original Projection 同样对 H12 应用目标层 input_layernorm，再投影；与旧版基线定义不同。
- 本次复用既有 validation/test split，未重新调参；不是全新独立确认集。Gate 1 不替代真实缓存续写门槛。

## 独立测试集结果

| Method | Added params | K NMSE | K cosine | V NMSE | V cosine | Attention NMSE | Attention cosine |
|:---|---:|---:|---:|---:|---:|---:|---:|
| zero | 0 | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 | 0.000000 |
| random_linear | 1,048,576 | 1.009590 | -0.007878 | 1.006745 | -0.027847 | 1.003541 | -0.041256 |
| original_projection | 0 | 0.357633 | 0.825730 | 0.279648 | 0.853234 | 0.267511 | 0.869492 |
| trained_linear | 1,048,576 | 0.238458 | 0.876381 | 0.351265 | 0.845246 | 0.504495 | 0.835466 |
| low_rank_64 | 196,608 | 0.395569 | 0.771584 | 0.485170 | 0.693050 | 0.294054 | 0.801150 |
| low_rank_128 | 393,216 | 0.322490 | 0.817528 | 0.389286 | 0.763964 | 0.179849 | 0.854851 |
| low_rank_256 | 786,432 | 0.257333 | 0.856667 | 0.290437 | 0.830247 | 0.138790 | 0.893455 |
| multi_layer_fusion | 1,048,582 | 0.276759 | 0.855767 | 0.384097 | 0.831815 | 0.487796 | 0.825806 |

## Gate 1

- Validation-selected method：`low_rank_256`。
- 测试集平均 K/V NMSE 相对 Original Projection 改善：14.05%。
- 测试集 Attention-output NMSE 相对改善：48.12%。
- Attention 改善的 paired bootstrap 95% CI：[47.24%, 48.93%]。

## 非对称融合权重

- K 对 H4/H8/H12 的权重：[0.24213427305221558, 0.36929482221603394, 0.3885708749294281]。
- V 对 H4/H8/H12 的权重：[0.2290930300951004, 0.38419145345687866, 0.38671553134918213]。

## 结论

Layer 15 的 K/V 可从浅层表示恢复，且功能误差显著优于 Original Projection 与随机基线。
