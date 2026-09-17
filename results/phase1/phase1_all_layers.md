# Qwen3.5-0.8B Phase 1 Upper Attention Recoverability

**Phase 1：GO**

Layers 15/19/23 均通过 Gate 1，可以进入 GDN State Recoverability。

## 跨层结果

| Layer | Validation-selected method | Original Attention NMSE | Selected Attention NMSE | Attention improvement | Bootstrap 95% CI | K/V improvement | Gate |
|---:|:---|---:|---:|---:|:---|---:|:---|
| 15 | low_rank_256 | 0.747987 | 0.051127 | 93.16% | [93.00%, 93.31%] | 80.51% | GO |
| 19 | multi_layer_fusion | 0.551331 | 0.075736 | 86.26% | [85.91%, 86.59%] | 73.95% | GO |
| 23 | low_rank_256 | 0.807416 | 0.023365 | 97.11% | [97.04%, 97.17%] | 83.90% | GO |

## 统一实验口径

- WikiText-103：每层 4,999,936 train tokens，validation/test 各 262,144 tokens。
- Source：H4/H8/H12；Target：各 attention layer 的 raw pre-RoPE K/V。
- Validation 选择方法，test 只用于最终 Gate；functional metric 固定对应层的 teacher Query。
- Gate 要求 K/V 与 Attention-output NMSE 均较 Original Projection 至少改善 10%，且 paired bootstrap 95% CI 下界大于 0。

## 决策

Phase 1 完成。三层均表明 shallow representation 可以恢复 upper-attention memory；下一里程碑为 Phase 2 单个上层 GDN recurrent/conv state probe。
