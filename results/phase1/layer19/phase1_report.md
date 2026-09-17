# Qwen3.5-0.8B Phase 1 Attention KV Recoverability

**Gate 1：GO**

验证集选择 `multi_layer_fusion`，其在独立测试集上同时改善 K/V reconstruction 与 teacher-query attention output。

## 实验设置

- Teacher：`/home/liu/CED/models/Qwen3.5-0.8B-Base`，revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`。
- Source：H4/H8/H12（decoder layer 3/7/11 输出）；Target：layer 19 raw pre-RoPE K/V。
- 训练：4,999,936 tokens，1,221 steps，452.1 秒。
- 序列长度：256；held-out validation/test 各 1,024 sequences。
- Functional metric：固定 teacher layer-19 Query，对 teacher/predicted K/V 应用原 KNorm 与 RoPE 后计算 causal attention output。

## 独立测试集结果

| Method | Added params | K NMSE | K cosine | V NMSE | V cosine | Attention NMSE | Attention cosine |
|:---|---:|---:|---:|---:|---:|---:|---:|
| zero | 0 | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 | 0.000000 |
| random_linear | 1,048,576 | 1.203233 | -0.000454 | 1.066935 | 0.004648 | 1.040556 | 0.008498 |
| original_projection | 0 | 0.530118 | 0.815931 | 0.582062 | 0.746381 | 0.551331 | 0.796932 |
| trained_linear | 1,048,576 | 0.111560 | 0.936329 | 0.183696 | 0.888264 | 0.077536 | 0.934499 |
| low_rank_64 | 196,608 | 0.192034 | 0.883605 | 0.313166 | 0.781402 | 0.139459 | 0.867860 |
| low_rank_128 | 393,216 | 0.148833 | 0.911958 | 0.244980 | 0.834987 | 0.110705 | 0.895954 |
| low_rank_256 | 786,432 | 0.120364 | 0.929960 | 0.190560 | 0.874445 | 0.085660 | 0.919403 |
| multi_layer_fusion | 1,048,582 | 0.114685 | 0.934568 | 0.175085 | 0.894872 | 0.075736 | 0.937797 |

## Gate 1

- Validation-selected method：`multi_layer_fusion`。
- 测试集平均 K/V NMSE 相对 Original Projection 改善：73.95%。
- 测试集 Attention-output NMSE 相对改善：86.26%。
- Attention 改善的 paired bootstrap 95% CI：[85.91%, 86.59%]。

## 非对称融合权重

- K 对 H4/H8/H12 的权重：[0.28961700201034546, 0.29817646741867065, 0.4122064709663391]。
- V 对 H4/H8/H12 的权重：[0.31152886152267456, 0.29078438878059387, 0.3976867198944092]。

## 结论

Layer 19 的 K/V 可从浅层表示恢复，且功能误差显著优于 Original Projection 与随机基线。
