# Qwen3.5-0.8B Phase 1 Attention KV Recoverability

**Gate 1：GO**

验证集选择 `low_rank_256`，其在独立测试集上同时改善 K/V reconstruction 与 teacher-query attention output。

## 实验设置

- Teacher：`/home/liu/CED/models/Qwen3.5-0.8B-Base`，revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`。
- Source：H4/H8/H12（decoder layer 3/7/11 输出）；Target：layer 19 raw pre-RoPE K/V。
- 训练：4,999,936 tokens，1,221 steps，447.8 秒。
- 序列长度：256；held-out validation/test 各 1,024 sequences。
- Functional metric：固定真实 teacher layer-19 Query，对 teacher/predicted K/V 应用原 KNorm 与 RoPE 后计算 causal attention output。
- Target 为 input_layernorm 之后的 self_attn 输入；首批目标经 KNorm/RoPE 后必须与真实缓存逐位相等。
- Original Projection 同样对 H12 应用目标层 input_layernorm，再投影；与旧版基线定义不同。
- 本次复用既有 validation/test split，未重新调参；不是全新独立确认集。Gate 1 不替代真实缓存续写门槛。

## 独立测试集结果

| Method | Added params | K NMSE | K cosine | V NMSE | V cosine | Attention NMSE | Attention cosine |
|:---|---:|---:|---:|---:|---:|---:|---:|
| zero | 0 | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 | 0.000000 |
| random_linear | 1,048,576 | 1.005568 | -0.000994 | 1.001140 | 0.005365 | 1.000003 | 0.010277 |
| original_projection | 0 | 0.447003 | 0.820856 | 0.456372 | 0.755088 | 0.261959 | 0.800013 |
| trained_linear | 1,048,576 | 0.206686 | 0.892894 | 0.476377 | 0.786776 | 0.351787 | 0.858763 |
| low_rank_64 | 196,608 | 0.318690 | 0.808038 | 0.469497 | 0.672496 | 0.194772 | 0.821242 |
| low_rank_128 | 393,216 | 0.260452 | 0.847241 | 0.413898 | 0.721126 | 0.175108 | 0.840904 |
| low_rank_256 | 786,432 | 0.207460 | 0.881352 | 0.346887 | 0.776521 | 0.150311 | 0.865673 |
| multi_layer_fusion | 1,048,582 | 0.227356 | 0.883134 | 0.509027 | 0.784411 | 0.401582 | 0.850883 |

## Gate 1

- Validation-selected method：`low_rank_256`。
- 测试集平均 K/V NMSE 相对 Original Projection 改善：38.64%。
- 测试集 Attention-output NMSE 相对改善：42.62%。
- Attention 改善的 paired bootstrap 95% CI：[41.66%, 43.67%]。

## 非对称融合权重

- K 对 H4/H8/H12 的权重：[0.25399282574653625, 0.35330119729042053, 0.3927059769630432]。
- V 对 H4/H8/H12 的权重：[0.22911499440670013, 0.3772501051425934, 0.39363494515419006]。

## 结论

Layer 19 的 K/V 可从浅层表示恢复，且功能误差显著优于 Original Projection 与随机基线。
