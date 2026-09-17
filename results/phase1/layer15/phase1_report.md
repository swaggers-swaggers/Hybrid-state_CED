# Qwen3.5-0.8B Phase 1 Attention KV Recoverability

**Gate 1：GO**

验证集选择 `low_rank_256`，其在独立测试集上同时改善 K/V reconstruction 与 teacher-query attention output。

## 实验设置

- Teacher：`/home/liu/CED/models/Qwen3.5-0.8B-Base`，revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`。
- Source：H4/H8/H12（decoder layer 3/7/11 输出）；Target：layer 15 raw pre-RoPE K/V。
- 训练：4,999,936 tokens，1,221 steps，451.3 秒。
- 序列长度：256；held-out validation/test 各 1,024 sequences。
- Functional metric：固定 teacher layer-19 Query，对 teacher/predicted K/V 应用原 KNorm 与 RoPE 后计算 causal attention output。

## 独立测试集结果

| Method | Added params | K NMSE | K cosine | V NMSE | V cosine | Attention NMSE | Attention cosine |
|:---|---:|---:|---:|---:|---:|---:|---:|
| zero | 0 | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 | 0.000000 |
| random_linear | 1,048,576 | 1.744971 | -0.007478 | 1.253464 | -0.027328 | 1.008537 | -0.019448 |
| original_projection | 0 | 0.424726 | 0.831482 | 0.488244 | 0.861374 | 0.747987 | 0.844877 |
| trained_linear | 1,048,576 | 0.090376 | 0.929743 | 0.121064 | 0.929606 | 0.173299 | 0.969533 |
| low_rank_64 | 196,608 | 0.171956 | 0.864651 | 0.200770 | 0.786984 | 0.082779 | 0.945023 |
| low_rank_128 | 393,216 | 0.125652 | 0.901767 | 0.126971 | 0.869556 | 0.064277 | 0.963705 |
| low_rank_256 | 786,432 | 0.097921 | 0.923407 | 0.080028 | 0.918384 | 0.051127 | 0.975459 |
| multi_layer_fusion | 1,048,582 | 0.102882 | 0.920473 | 0.097679 | 0.926312 | 0.107411 | 0.974007 |

## Gate 1

- Validation-selected method：`low_rank_256`。
- 测试集平均 K/V NMSE 相对 Original Projection 改善：80.51%。
- 测试集 Attention-output NMSE 相对改善：93.16%。
- Attention 改善的 paired bootstrap 95% CI：[93.00%, 93.31%]。

## 非对称融合权重

- K 对 H4/H8/H12 的权重：[0.27413153648376465, 0.33796149492263794, 0.3879069983959198]。
- V 对 H4/H8/H12 的权重：[0.27829062938690186, 0.44072282314300537, 0.2809865474700928]。

## 结论

Layer 15 的 K/V 可从浅层表示恢复，且功能误差显著优于 Original Projection 与随机基线。
