# Qwen3.5-0.8B Phase 1 Attention KV Recoverability

**Gate 1：GO**

验证集选择 `low_rank_256`，其在独立测试集上同时改善 K/V reconstruction 与 teacher-query attention output。

## 实验设置

- Teacher：`/home/liu/CED/models/Qwen3.5-0.8B-Base`，revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`。
- Source：H4/H8/H12（decoder layer 3/7/11 输出）；Target：layer 23 raw pre-RoPE K/V。
- 训练：4,999,936 tokens，1,221 steps，451.8 秒。
- 序列长度：256；held-out validation/test 各 1,024 sequences。
- Functional metric：固定 teacher layer-19 Query，对 teacher/predicted K/V 应用原 KNorm 与 RoPE 后计算 causal attention output。

## 独立测试集结果

| Method | Added params | K NMSE | K cosine | V NMSE | V cosine | Attention NMSE | Attention cosine |
|:---|---:|---:|---:|---:|---:|---:|---:|
| zero | 0 | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 | 0.000000 |
| random_linear | 1,048,576 | 1.116126 | 0.002612 | 1.004531 | 0.000318 | 1.002137 | -0.004145 |
| original_projection | 0 | 0.703721 | 0.689014 | 0.827601 | 0.831129 | 0.807416 | 0.919454 |
| trained_linear | 1,048,576 | 0.179701 | 0.902245 | 0.057156 | 0.980645 | 0.027787 | 0.990915 |
| low_rank_64 | 196,608 | 0.287319 | 0.833125 | 0.067813 | 0.967327 | 0.033462 | 0.984057 |
| low_rank_128 | 393,216 | 0.236127 | 0.865579 | 0.059420 | 0.971258 | 0.028928 | 0.986254 |
| low_rank_256 | 786,432 | 0.196377 | 0.890675 | 0.050105 | 0.975778 | 0.023365 | 0.988967 |
| multi_layer_fusion | 1,048,582 | 0.181851 | 0.901333 | 0.072328 | 0.977862 | 0.038527 | 0.988761 |

## Gate 1

- Validation-selected method：`low_rank_256`。
- 测试集平均 K/V NMSE 相对 Original Projection 改善：83.90%。
- 测试集 Attention-output NMSE 相对改善：97.11%。
- Attention 改善的 paired bootstrap 95% CI：[97.04%, 97.17%]。

## 非对称融合权重

- K 对 H4/H8/H12 的权重：[0.2923319637775421, 0.2987270653247833, 0.40894097089767456]。
- V 对 H4/H8/H12 的权重：[0.2485593557357788, 0.3630008399486542, 0.38843977451324463]。

## 结论

Layer 23 的 K/V 可从浅层表示恢复，且功能误差显著优于 Original Projection 与随机基线。
