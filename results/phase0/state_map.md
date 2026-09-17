# Qwen3.5-0.8B Phase 0 Inference-State Map

**验收结论：PASS**

完整 inference-state map 与真实 cache 重注入门禁均已通过。

## 运行环境

- timestamp: `2026-09-16T18:55:44.234883+08:00`
- model_path: `/home/liu/CED/models/Qwen3.5-0.8B-Base`
- model_revision: `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`
- weights_etag_sha256: `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c`
- torch: `2.11.0+cu130`
- transformers: `5.12.1`
- device: `cuda:0`
- gpu: `NVIDIA GeForce RTX 5060 Ti`
- driver: `595.84`
- attention_backend: `sdpa`

## 模型结构

- 层数：24（18 个 GDN + 6 个 full attention）
- Hidden size：1024
- Full-attention 层索引：[3, 7, 11, 15, 19, 23]
- 原生上下文长度：262144

## 状态语义与依赖

- `H_l`：每层输入/输出均为 `[1, N, 1024]` BF16，随序列长度线性增长，依赖上一层 residual stream。
- Attention：raw K、KNorm 后 K 与 V 均被 hook 捕获；最终 K/V cache 为 `[1, 2, N, 256]` BF16，依赖隐藏状态投影、KNorm、RoPE 与按位置拼接。
- GDN：recurrent state 为 `[1, 16, 128, 128]` BF16，conv state 为 `[1, 6144, 4]` BF16；二者大小不随 N 增长，分别汇总 prefix recurrence 与最近 4 个卷积输入。

## 测量口径

- batch size 1，BF16，固定伪随机 token；每个长度预热 1 次、采样 3 次并报告中位数。
- Prefill/decode 时延覆盖 24 层 text backbone 与 cache 更新，不含 tokenizer、vision tower、LM head 和 cache 克隆。
- GDN prefill backend：`transformers.models.qwen3_5.modeling_qwen3_5.torch_chunk_gated_delta_rule`。
- GDN decode backend：`transformers.models.qwen3_5.modeling_qwen3_5.torch_recurrent_gated_delta_rule`。
- Causal-conv prefill/decode：`none` / `transformers.models.qwen3_5.modeling_qwen3_5.torch_causal_conv1d_update`。

## 运行时测量

| Context | Cache MiB | Peak VRAM MiB | Prefill ms | Decode ms | 状态 |
|---:|---:|---:|---:|---:|:---|
| 512 | 15.84 | 1732.54 | 62.837 | 14.384 | ok |
| 2048 | 33.84 | 1895.45 | 173.444 | 13.823 | ok |
| 8192 | 105.84 | 2545.23 | 870.599 | 13.277 | ok |

## 逐层运行时状态

下表使用最长的成功测量上下文。

| Layer | Type | 实测张量 | Cache bytes | Prefill ms | Decode ms |
|---:|:---|:---|---:|---:|---:|
| 0 | gdn | `hidden_input=[1, 8192, 1024] bfloat16, hidden_output=[1, 8192, 1024] bfloat16; conv=[1, 6144, 4] bfloat16, recurrent=[1, 16, 128, 128] bfloat16` | 573440 | 41.224 | 0.596 |
| 1 | gdn | `hidden_input=[1, 8192, 1024] bfloat16, hidden_output=[1, 8192, 1024] bfloat16; conv=[1, 6144, 4] bfloat16, recurrent=[1, 16, 128, 128] bfloat16` | 573440 | 41.649 | 0.599 |
| 2 | gdn | `hidden_input=[1, 8192, 1024] bfloat16, hidden_output=[1, 8192, 1024] bfloat16; conv=[1, 6144, 4] bfloat16, recurrent=[1, 16, 128, 128] bfloat16` | 573440 | 42.075 | 0.553 |
| 3 | attention_kv | `hidden_input=[1, 8192, 1024] bfloat16, raw_k=[1, 8192, 512] bfloat16, knorm_k=[1, 8192, 2, 256] bfloat16, raw_v=[1, 8192, 512] bfloat16, hidden_output=[1, 8192, 1024] bfloat16; keys=[1, 2, 8192, 256] bfloat16, values=[1, 2, 8192, 256] bfloat16` | 16777216 | 19.834 | 0.471 |
| 4 | gdn | `hidden_input=[1, 8192, 1024] bfloat16, hidden_output=[1, 8192, 1024] bfloat16; conv=[1, 6144, 4] bfloat16, recurrent=[1, 16, 128, 128] bfloat16` | 573440 | 42.075 | 0.542 |
| 5 | gdn | `hidden_input=[1, 8192, 1024] bfloat16, hidden_output=[1, 8192, 1024] bfloat16; conv=[1, 6144, 4] bfloat16, recurrent=[1, 16, 128, 128] bfloat16` | 573440 | 41.276 | 0.548 |
| 6 | gdn | `hidden_input=[1, 8192, 1024] bfloat16, hidden_output=[1, 8192, 1024] bfloat16; conv=[1, 6144, 4] bfloat16, recurrent=[1, 16, 128, 128] bfloat16` | 573440 | 41.972 | 0.583 |
| 7 | attention_kv | `hidden_input=[1, 8192, 1024] bfloat16, raw_k=[1, 8192, 512] bfloat16, knorm_k=[1, 8192, 2, 256] bfloat16, raw_v=[1, 8192, 512] bfloat16, hidden_output=[1, 8192, 1024] bfloat16; keys=[1, 2, 8192, 256] bfloat16, values=[1, 2, 8192, 256] bfloat16` | 16777216 | 19.491 | 0.545 |
| 8 | gdn | `hidden_input=[1, 8192, 1024] bfloat16, hidden_output=[1, 8192, 1024] bfloat16; conv=[1, 6144, 4] bfloat16, recurrent=[1, 16, 128, 128] bfloat16` | 573440 | 42.079 | 0.553 |
| 9 | gdn | `hidden_input=[1, 8192, 1024] bfloat16, hidden_output=[1, 8192, 1024] bfloat16; conv=[1, 6144, 4] bfloat16, recurrent=[1, 16, 128, 128] bfloat16` | 573440 | 41.603 | 0.539 |
| 10 | gdn | `hidden_input=[1, 8192, 1024] bfloat16, hidden_output=[1, 8192, 1024] bfloat16; conv=[1, 6144, 4] bfloat16, recurrent=[1, 16, 128, 128] bfloat16` | 573440 | 41.331 | 0.568 |
| 11 | attention_kv | `hidden_input=[1, 8192, 1024] bfloat16, raw_k=[1, 8192, 512] bfloat16, knorm_k=[1, 8192, 2, 256] bfloat16, raw_v=[1, 8192, 512] bfloat16, hidden_output=[1, 8192, 1024] bfloat16; keys=[1, 2, 8192, 256] bfloat16, values=[1, 2, 8192, 256] bfloat16` | 16777216 | 20.231 | 0.490 |
| 12 | gdn | `hidden_input=[1, 8192, 1024] bfloat16, hidden_output=[1, 8192, 1024] bfloat16; conv=[1, 6144, 4] bfloat16, recurrent=[1, 16, 128, 128] bfloat16` | 573440 | 41.163 | 0.257 |
| 13 | gdn | `hidden_input=[1, 8192, 1024] bfloat16, hidden_output=[1, 8192, 1024] bfloat16; conv=[1, 6144, 4] bfloat16, recurrent=[1, 16, 128, 128] bfloat16` | 573440 | 42.123 | 0.538 |
| 14 | gdn | `hidden_input=[1, 8192, 1024] bfloat16, hidden_output=[1, 8192, 1024] bfloat16; conv=[1, 6144, 4] bfloat16, recurrent=[1, 16, 128, 128] bfloat16` | 573440 | 41.079 | 0.545 |
| 15 | attention_kv | `hidden_input=[1, 8192, 1024] bfloat16, raw_k=[1, 8192, 512] bfloat16, knorm_k=[1, 8192, 2, 256] bfloat16, raw_v=[1, 8192, 512] bfloat16, hidden_output=[1, 8192, 1024] bfloat16; keys=[1, 2, 8192, 256] bfloat16, values=[1, 2, 8192, 256] bfloat16` | 16777216 | 19.701 | 0.476 |
| 16 | gdn | `hidden_input=[1, 8192, 1024] bfloat16, hidden_output=[1, 8192, 1024] bfloat16; conv=[1, 6144, 4] bfloat16, recurrent=[1, 16, 128, 128] bfloat16` | 573440 | 41.737 | 0.555 |
| 17 | gdn | `hidden_input=[1, 8192, 1024] bfloat16, hidden_output=[1, 8192, 1024] bfloat16; conv=[1, 6144, 4] bfloat16, recurrent=[1, 16, 128, 128] bfloat16` | 573440 | 41.803 | 0.548 |
| 18 | gdn | `hidden_input=[1, 8192, 1024] bfloat16, hidden_output=[1, 8192, 1024] bfloat16; conv=[1, 6144, 4] bfloat16, recurrent=[1, 16, 128, 128] bfloat16` | 573440 | 42.059 | 0.542 |
| 19 | attention_kv | `hidden_input=[1, 8192, 1024] bfloat16, raw_k=[1, 8192, 512] bfloat16, knorm_k=[1, 8192, 2, 256] bfloat16, raw_v=[1, 8192, 512] bfloat16, hidden_output=[1, 8192, 1024] bfloat16; keys=[1, 2, 8192, 256] bfloat16, values=[1, 2, 8192, 256] bfloat16` | 16777216 | 19.354 | 0.474 |
| 20 | gdn | `hidden_input=[1, 8192, 1024] bfloat16, hidden_output=[1, 8192, 1024] bfloat16; conv=[1, 6144, 4] bfloat16, recurrent=[1, 16, 128, 128] bfloat16` | 573440 | 41.976 | 0.548 |
| 21 | gdn | `hidden_input=[1, 8192, 1024] bfloat16, hidden_output=[1, 8192, 1024] bfloat16; conv=[1, 6144, 4] bfloat16, recurrent=[1, 16, 128, 128] bfloat16` | 573440 | 41.419 | 0.538 |
| 22 | gdn | `hidden_input=[1, 8192, 1024] bfloat16, hidden_output=[1, 8192, 1024] bfloat16; conv=[1, 6144, 4] bfloat16, recurrent=[1, 16, 128, 128] bfloat16` | 573440 | 41.777 | 0.541 |
| 23 | attention_kv | `hidden_input=[1, 8192, 1024] bfloat16, raw_k=[1, 8192, 512] bfloat16, knorm_k=[1, 8192, 2, 256] bfloat16, raw_v=[1, 8192, 512] bfloat16, hidden_output=[1, 8192, 1024] bfloat16; keys=[1, 2, 8192, 256] bfloat16, values=[1, 2, 8192, 256] bfloat16` | 16777216 | 19.440 | 0.475 |

## 验证结果

- cache_roundtrip: **PASS** — max_abs=0.0, top1_equal=True
- attention_cache_injection: **PASS** — 12 K/V tensors copied exactly; functional continuation=PASS
- gdn_cache_injection: **PASS** — 36 recurrent/conv tensors copied exactly; functional continuation=PASS
- boundary: **PASS** — max_abs=0.1875, top1_equal=True
- causal: **PASS** — unchanged prefix max_abs=0.0
- backend_parity: **PASS** — prefill/decode functions remained unchanged
