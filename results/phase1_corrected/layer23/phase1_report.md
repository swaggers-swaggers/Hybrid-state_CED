# Qwen3.5-0.8B Phase 1 Attention KV Recoverability

**Gate 1：GO**

验证集选择 `low_rank_256`，其在独立测试集上同时改善 K/V reconstruction 与 teacher-query attention output。

## 实验设置

- Teacher：`/home/liu/CED/models/Qwen3.5-0.8B-Base`，revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`。
- Source：H4/H8/H12（decoder layer 3/7/11 输出）；Target：layer 23 raw pre-RoPE K/V。
- 训练：4,999,936 tokens，1,221 steps，452.3 秒。
- 序列长度：256；held-out validation/test 各 1,024 sequences。
- Functional metric：固定真实 teacher layer-23 Query，对 teacher/predicted K/V 应用原 KNorm 与 RoPE 后计算 causal attention output。
- Target 为 input_layernorm 之后的 self_attn 输入；首批目标经 KNorm/RoPE 后必须与真实缓存逐位相等。
- Original Projection 同样对 H12 应用目标层 input_layernorm，再投影；与旧版基线定义不同。
- 本次复用既有 validation/test split，未重新调参；不是全新独立确认集。Gate 1 不替代真实缓存续写门槛。

## 独立测试集结果

| Method | Added params | K NMSE | K cosine | V NMSE | V cosine | Attention NMSE | Attention cosine |
|:---|---:|---:|---:|---:|---:|---:|---:|
| zero | 0 | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 | 0.000000 |
| random_linear | 1,048,576 | 1.006578 | 0.003208 | 1.000298 | 0.000160 | 1.000221 | -0.004840 |
| original_projection | 0 | 0.872798 | 0.701738 | 0.396285 | 0.826741 | 0.315626 | 0.917999 |
| trained_linear | 1,048,576 | 0.249624 | 0.866878 | 0.363755 | 0.927327 | 0.334966 | 0.940814 |
| low_rank_64 | 196,608 | 0.385595 | 0.770627 | 0.084597 | 0.960214 | 0.040973 | 0.980399 |
| low_rank_128 | 393,216 | 0.319341 | 0.814240 | 0.077259 | 0.963108 | 0.037027 | 0.982217 |
| low_rank_256 | 786,432 | 0.263207 | 0.849817 | 0.069013 | 0.966787 | 0.033002 | 0.984141 |
| multi_layer_fusion | 1,048,582 | 0.265445 | 0.859261 | 0.432914 | 0.914362 | 0.399547 | 0.929778 |

## Gate 1

- Validation-selected method：`low_rank_256`。
- 测试集平均 K/V NMSE 相对 Original Projection 改善：73.82%。
- 测试集 Attention-output NMSE 相对改善：89.54%。
- Attention 改善的 paired bootstrap 95% CI：[89.28%, 89.79%]。

## 非对称融合权重

- K 对 H4/H8/H12 的权重：[0.25697454810142517, 0.3512186110019684, 0.39180678129196167]。
- V 对 H4/H8/H12 的权重：[0.22401976585388184, 0.38068369030952454, 0.39529654383659363]。

## 结论

Layer 23 的 K/V 可从浅层表示恢复，且功能误差显著优于 Original Projection 与随机基线。
