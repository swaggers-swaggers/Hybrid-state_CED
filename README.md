# Hybrid-State CED · Qwen3.5-0.8B

本仓库是《Hybrid-State CED-Qwen3.5-0.8B 实验计划书 v2.0》的代码实现。

研究目标：对**已完成预训练**的 Qwen3.5-0.8B 做结构 retrofit，在尽量冻结原模型参数的条件下，用较浅层的表示重建上层 decoder 在 decode 开始前所需的历史推理状态（heterogeneous inference-state reconstruction），从而跳过上层 prompt prefill、降低长上下文 TTFT。

与从头训练 CED 模型不同，本项目的核心问题不是"预测某个张量"，而是判断重建出的状态能否**功能上**替代 teacher 状态并支持稳定续写。每个阶段都设 Go/No-Go 门禁。

## 研究对象

Qwen3.5-0.8B-Base 文本主干，24 层，hidden size 1024，由 6 个重复组构成，每组 `3 × Gated DeltaNet + 1 × gated full-attention`：

| 属性 | 值 |
|:---|:---|
| Full attention 层（0-based） | 3, 7, 11, 15, 19, 23 |
| Gated DeltaNet 层 | 其余 18 层 |
| Full attention | 8 Q heads / 2 KV heads / head_dim 256 |
| Gated DeltaNet | 16 heads / head_dim 128，causal conv kernel 4 |
| Native context | 262,144 |

后续主切分为 layer 11 之后的 12/12：layers 0–11 为 Causal Encoder（产生 `H_E = H_12`），layers 12–23 为 Decoder（prompt 阶段尽量跳过）。

## 当前进度

| 阶段 | 内容 | 状态 |
|:---|:---|:---|
| Phase 0 | 完整 Inference-State Map 与 cache 抽注重注入验证 | **PASS** |
| Phase 1 | Upper Attention K/V Recoverability（layers 15/19/23） | **已修复并重训；三层 Gate 1 GO** |
| K/V continuation pilot | 新版权重，256/2048 长度、128 步续写 | **联合续写仍 HOLD；暂不进入 GDN** |
| Phase 2 | Gated DeltaNet State Recoverability | 未开始 |
| Phase 3–6 | Cache Injection / Partial CED / Full Hybrid-State CED / 低秩与 kernel | 未开始 |

2026-09-17 的首次真实缓存注入 pilot 未通过推进门槛。审计发现旧版 Phase 1 在 input_layernorm 之前捕获目标 hidden，而实际 self_attn 使用归一化之后的输入。当前代码已改为捕获真实 self_attn 输入，并在训练和每个评估 split 的首批数据上强制验证 K/V 与真实缓存逐位一致。新版 checkpoint 带目标口径版本，评估拒绝混用旧权重。Original Projection 基线同样使用目标层输入归一化。旧结果仅供审计，见 [首次实验报告](results/kv_continuation_pilot/report.md) 与 [口径审计](results/kv_continuation_pilot/protocol_and_audit.md)。

完整修复复验流程（校验已有数据、GPU 回归测试、三层全量重训与评估、续写复验、最终测试）：

```bash
conda run -n ai python scripts/run_phase1_corrected.py
```

所有新版产物写入 `results/phase1_corrected/` 和 `checkpoints/phase1_corrected/`，原版不覆盖。续写复验使用不与首次 pilot 重叠的窗口，仍属于既有 test 语料，不能当作全新独立确认集；方法只依照 validation 选择，不按续写结果调参。

修复复验已完成：三层新版 Attention NMSE 相对归一化原始投影分别改善 **48.12%、42.62%、89.54%**，均通过 Gate 1；但联合续写平均 KL 为 **0.1115 / 0.2930**，高于同批窗口原始投影的 **0.0897 / 0.1859**，所以继续暂缓 GDN。全部 **21 项测试通过（含真实 GPU，无跳过）**。详见 [修复与复验结论](results/phase1_corrected/repair_summary.md)。

Phase 1 历史代理目标在 test split 上，K/V 与 attention-output NMSE 相对 Original Projection 的改善：

| Layer | Validation-selected | Attention NMSE 改善 | 95% CI | K/V NMSE 改善 |
|---:|:---|---:|:---|---:|
| 15 | low_rank_256 | 93.16% | [93.00%, 93.31%] | 80.51% |
| 19 | multi_layer_fusion | 86.26% | [85.91%, 86.59%] | 73.95% |
| 23 | low_rank_256 | 97.11% | [97.04%, 97.17%] | 83.90% |

Phase 0 实测（batch 1、BF16、RTX 5060 Ti、`sdpa`）：

| Context | Cache MiB | Peak VRAM MiB | Prefill ms | Decode ms |
|---:|---:|---:|---:|---:|
| 512 | 15.84 | 1732.54 | 62.837 | 14.384 |
| 2048 | 33.84 | 1895.45 | 173.444 | 13.823 |
| 8192 | 105.84 | 2545.23 | 870.599 | 13.277 |

每层 cache：full-attention 为 `[1, 2, N, 256]` K/V，随 N 线性增长（8192 时 16 MiB/层）；GDN 为 conv `[1, 6144, 4]` + recurrent `[1, 16, 128, 128]`，共 560 KiB/层且不随 N 增长。

## 环境

固定使用现有 Conda 环境 `ai`；模型位于 `models/Qwen3.5-0.8B-Base`（`.gitignore` 中忽略，可按官方 checkpoint 重新下载）。

```bash
conda run -n ai python scripts/check_environment.py
```

该脚本检查 python/torch/transformers 版本、CUDA 可用性、GPU 与模型文件是否存在；模型或 CUDA 缺失时返回非 0。

## 目录结构

```text
configs/     Phase 0 / Phase 1 的实验配置（长度、容差、超参、Gate 阈值）
phase0/      Phase 0 库：静态 state map、异构 cache 抽注、对比与报告
phase1/      Phase 1 库：K/V probe 模型、teacher 特征捕获、指标、IO、报告
scripts/     可执行入口：环境检查、state map trace、数据准备、训练、评估、汇总
tests/       单元测试与（可选的）真实模型 GPU 测试
data/         打包好的定长 WikiText-103 token shards 与 manifest
results/      Phase 0 / Phase 1 的实验产物（json / md / png / jsonl）
checkpoints/  训练好的 K/V probe 权重
models/       官方 Qwen3.5-0.8B-Base 权重与 tokenizer（gitignore）
.cache/       本地 HF_HOME（数据集与权重下载缓存，gitignore）
.deps/        vendored 依赖（socksio），供数据流式下载走代理使用
```

## Phase 0：Inference-State Map

产出完整状态地图，并验证真实 cache 的"抽取 → 重新注入"链路可信，这是后续一切"预测 cache 注入"实验的前提。

```bash
./scripts/run_phase0.sh
```

等价于：环境检查 → `scripts/trace_state_map.py` → 带真实模型的单元测试。

`trace_state_map.py` 依次完成：

1. 由官方 config 静态推导逐层 H/K/V/conv/recurrent 的 shape、dtype、bytes、增长规律与依赖，并与期望结构比对；
2. 对每个 context 长度做 prefill/decode profiling，hook 捕获逐层张量与 CUDA Event 逐层时延，记录峰值显存；
3. **cache round-trip**：从 teacher cache 重建新的 `DynamicCache`（attn K/V 与 GDN conv/recurrent state 分路径复制），断言存储独立、张量逐位相等，并用同一 next token 比对 logits；
4. **boundary**：N-1 prefix + N boundary token 路径 vs 完整 forward，比对 top-1、max_abs、KL、cosine；
5. **causal**：修改后半段输入，确认前半段 hidden 不变；
6. **backend parity**：前后两次 backend fingerprint 一致（prefill/decode 使用的 GDN、causal-conv 实现未变）。

输出到 `results/phase0/`：

- `state_map.json` — 机器可读的完整状态地图、运行环境、逐层测量与实验元数据；
- `state_map.md` — 逐层状态、内存、时延与验收结论；
- `verification.json` — round-trip / injection / boundary / causal / parity 的数值结果。

显存繁忙时可先做短序列 smoke test：

```bash
conda run -n ai python scripts/trace_state_map.py \
  --config configs/qwen35_08b_state_map.yaml \
  --context-lengths 64 --profile-runs 1 --warmup-runs 0
```

### 判定标准

Phase 0 只有在以下条件全部满足时才标记 `PASS`：

1. 24 层类型与官方配置一致（18 GDN + 6 full attention，索引 `[3,7,11,15,19,23]`）；
2. 所有指定 context length 均完成状态与性能采集；
3. cache clone 后 single-token logits 最大绝对误差不超过阈值且 top-1 一致，且 K/V 与 GDN state 张量逐位相等；
4. boundary 路径 top-1 一致且误差/KL/cosine 满足阈值；
5. 因果测试通过；
6. prefill 与 decode 使用相同 backend fingerprint。

任何 OOM、GPU 被占用、缺失长度或 backend 差异都会记录为未通过，**不以理论值替代实测值**。

## Phase 1：Attention K/V Recoverability

固定 Qwen3.5 teacher，用 H4/H8/H12（decoder layer 3/7/11 输出）构造上层 attention 层的 raw pre-RoPE K/V，比较六类方法：

- `zero`：全零 K/V（下界）；
- `random_linear`：随机初始化同规模线性层；
- `original_projection`：对最深 source 使用目标层 `input_layernorm`，再应用原始 `k_proj`/`v_proj`；
- `trained_linear`：训练 full-rank `P_K`/`P_V`；
- `low_rank_64/128/256`：`P = A·B` 低秩分解；
- `multi_layer_fusion`：K 与 V 各自用 softmax 权重融合 H4/H8/H12（对应假设 H2：K/V 最优信息来源不同）。

训练数据为 WikiText-103 官方 split，按 split 独立打包成定长 256 的序列，train 约 5M tokens、validation/test 各 262K tokens。Gate 1 在 validation 上选方法，再在 test 上报 K/V NMSE、cosine 与固定 teacher Query 下的 attention-output NMSE，并对 functional improvement 做 paired bootstrap。

```bash
./scripts/run_phase1.sh
```

该脚本会准备数据（`PYTHONPATH=.deps`、`HF_HOME=.cache/huggingface`），对 layers **19 / 15 / 23** 依次训练与评估，最后汇总并运行真实模型测试。已有数据与 checkpoint 时可分别执行：

```bash
conda run -n ai python scripts/train_kv_probe.py    --config configs/qwen35_08b_phase1.yaml
conda run -n ai python scripts/evaluate_kv_probe.py --config configs/qwen35_08b_phase1.yaml
```

主要输出：

- `results/phase1_corrected/layer{15,19,23}/`：单层报告、JSON、图表、per-batch 指标和训练日志；
- `results/phase1_corrected/phase1_all_layers.md` / `.json` 与 `phase1_layer_depth.png`：跨层汇总；
- `checkpoints/phase1_corrected/layer{15,19,23}_kv_probes.pt`：新版权重；
- `results/phase1_corrected/continuation/`：完整修复流程额外生成的真实续写复验；
- `results/phase1_corrected/run_stages.json`：各阶段退出状态和耗时。

### Gate 1

测试集上，validation 选出的方法必须同时满足：

1. 平均 K/V NMSE 相对 `original_projection` 改善 ≥ 10%；
2. attention-output NMSE 相对 `original_projection` 改善 ≥ 10%；
3. attention 改善的 paired bootstrap 95% CI 下界 > 0；
4. attention-output NMSE 优于 `random_linear`。

三层均 GO 只代表通过单层重建门槛；是否进入 GDN State Recoverability 还须结合真实缓存联合替换与续写门槛。修复代码不等于实验成功。

## K/V 缓存注入小规模续写实验

```bash
conda run -n ai python scripts/run_kv_continuation_pilot.py
```

固定配置位于 `configs/kv_continuation_pilot.json`。256/2048 长度各 16 个不重叠测试窗口，每个窗口 128 步同 token 续写；保留 teacher GDN 状态，比较单层替换、三层替换、original projection、zero K/V 与真实缓存重编码对照。输出为 `results/kv_continuation_pilot/results.json` 和 `report.md`，包括逐窗口逐步数值、checkpoint 哈希、配对 bootstrap 与门槛判定。它不代表自由生成、完整 CED 或速度 benchmark。

首次旧权重 pilot 三层预测的平均 KL 分别为 0.2678、0.4617；旧 original projection 为 0.2422、0.4653，判定 **HOLD_REPAIR_KV**。修复后的 pilot 改用新版权重、归一化的原始投影基线及不同窗口，不能把两轮数值差异完全归因于修复。新版复验由上面的 `run_phase1_corrected.py` 执行；默认 `run_kv_continuation_pilot.py` 仍用于重现首次旧权重实验。

## 测试

无需额外安装 pytest：

```bash
conda run -n ai python -m unittest discover -s tests -v
```

默认只检查纯 Python/CPU 逻辑（静态 state map、CPU 上的 cache clone 独立性、probe 形状与参数量、指标与 gate 逻辑、已产出的 Phase 1 结果与数据 manifest）。设置 `CED_RUN_MODEL_TESTS=1` 后额外加载真实模型执行 GPU 测试（cache round-trip、boundary、因果性）：

```bash
CED_RUN_MODEL_TESTS=1 conda run -n ai python -m unittest discover -s tests -v
```

## 约定

- **不做理论值替身**：缺测量、OOM、backend 不一致一律记为未通过，并保留原始错误信息作为证据。
- **teacher 全程冻结**：Phase 1 只训练 probe，probe 初始化在 teacher 加载后重新设种子以保证可复现。
- **train/validation/test 严格隔离**：上游 split 保留，序列只在 split 内部打包；方法选择只用 validation，test 只用于最终 Gate。
- **功能指标优先**：state 数值误差是辅助，attention-output NMSE（以及后续阶段的 logits/continuation 误差）才是判据。
- **可复现性**：结果中记录模型 revision、权重 etag、torch/transformers 版本、GPU 与 backend fingerprint。
