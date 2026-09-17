# Qwen3.5-0.8B Phase 1 Upper Attention Recoverability

**Phase 1：GO**

Layers 15/19/23 均通过 Gate 1；是否进入 GDN 仍取决于真实缓存续写复验。

## 跨层结果

| Layer | Validation-selected method | Original Attention NMSE | Selected Attention NMSE | Attention improvement | Bootstrap 95% CI | K/V improvement | Gate |
|---:|:---|---:|---:|---:|:---|---:|:---|
| 15 | low_rank_256 | 0.267511 | 0.138790 | 48.12% | [47.24%, 48.93%] | 14.05% | GO |
| 19 | low_rank_256 | 0.261959 | 0.150311 | 42.62% | [41.66%, 43.67%] | 38.64% | GO |
| 23 | low_rank_256 | 0.315626 | 0.033002 | 89.54% | [89.28%, 89.79%] | 73.82% | GO |

## 统一实验口径

- WikiText-103：每层 4,999,936 train tokens，validation/test 各 262,144 tokens。
- Source：H4/H8/H12；Target：各 attention layer 在 input_layernorm 之后的真实 raw pre-RoPE K/V。
- Original Projection 对 H12 施加目标层 input_layernorm 后再投影；旧版结果与新版基线不能直接作同口径百分比对比。
- 首批训练/评估目标均需与真实缓存逐位相等；本次复用既有数据 split，未使用新的独立确认集。
- Validation 选择方法，test 只用于最终 Gate；functional metric 固定对应层的 teacher Query。
- Gate 要求 K/V 与 Attention-output NMSE 均较 Original Projection 至少改善 10%，且 paired bootstrap 95% CI 下界大于 0。

## 决策

修正目标后的三层 probe 均通过 Gate 1。先检查真实缓存联合替换与续写结果，通过后再进入单层 GDN pilot。
