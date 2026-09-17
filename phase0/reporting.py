from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _mib(value: int | float) -> str:
    return f"{value / 2**20:.2f}"


def render_markdown(result: dict[str, Any]) -> str:
    verdict = result["verdict"]
    lines = [
        "# Qwen3.5-0.8B Phase 0 Inference-State Map",
        "",
        f"**验收结论：{verdict['status']}**",
        "",
        "完整 inference-state map 与真实 cache 重注入门禁均已通过。"
        if verdict["status"] == "PASS"
        else "Phase 0 尚有验收项未通过。",
        "",
        "## 运行环境",
        "",
    ]
    env = result["environment"]
    for key in (
        "timestamp",
        "model_path",
        "model_revision",
        "weights_etag_sha256",
        "torch",
        "transformers",
        "device",
        "gpu",
        "driver",
        "attention_backend",
    ):
        lines.append(f"- {key}: `{env.get(key, 'unknown')}`")

    architecture = result["static_state_map"]["architecture"]
    lines.extend(
        [
            "",
            "## 模型结构",
            "",
            f"- 层数：{architecture['num_hidden_layers']}（{architecture['linear_attention_layers']} 个 GDN + {architecture['full_attention_layers']} 个 full attention）",
            f"- Hidden size：{architecture['hidden_size']}",
            f"- Full-attention 层索引：{architecture['full_attention_indices']}",
            f"- 原生上下文长度：{architecture['native_context']}",
            "",
            "## 状态语义与依赖",
            "",
            "- `H_l`：每层输入/输出均为 `[1, N, 1024]` BF16，随序列长度线性增长，依赖上一层 residual stream。",
            "- Attention：raw K、KNorm 后 K 与 V 均被 hook 捕获；最终 K/V cache 为 `[1, 2, N, 256]` BF16，依赖隐藏状态投影、KNorm、RoPE 与按位置拼接。",
            "- GDN：recurrent state 为 `[1, 16, 128, 128]` BF16，conv state 为 `[1, 6144, 4]` BF16；二者大小不随 N 增长，分别汇总 prefix recurrence 与最近 4 个卷积输入。",
            "",
            "## 测量口径",
            "",
            f"- batch size 1，BF16，固定伪随机 token；每个长度预热 {result['settings']['warmup_runs']} 次、采样 {result['settings']['profile_runs']} 次并报告中位数。",
            "- Prefill/decode 时延覆盖 24 层 text backbone 与 cache 更新，不含 tokenizer、vision tower、LM head 和 cache 克隆。",
            f"- GDN prefill backend：`{env['backend_fingerprint']['gdn_prefill']}`。",
            f"- GDN decode backend：`{env['backend_fingerprint']['gdn_decode']}`。",
            f"- Causal-conv prefill/decode：`{env['backend_fingerprint']['causal_conv_prefill']}` / `{env['backend_fingerprint']['causal_conv_decode']}`。",
            "",
            "## 运行时测量",
            "",
            "| Context | Cache MiB | Peak VRAM MiB | Prefill ms | Decode ms | 状态 |",
            "|---:|---:|---:|---:|---:|:---|",
        ]
    )
    for run in result["runs"]:
        if run["status"] == "ok":
            lines.append(
                f"| {run['context_length']} | {_mib(run['cache_bytes'])} | {run['peak_vram_bytes']/2**20:.2f} | "
                f"{run['prefill_ms']['median']:.3f} | {run['decode_ms']['median']:.3f} | ok |"
            )
        else:
            lines.append(f"| {run['context_length']} | - | - | - | - | {run['status']}: {run.get('error', '')} |")

    lines.extend(
        [
            "",
            "## 逐层运行时状态",
            "",
            "下表使用最长的成功测量上下文。",
            "",
            "| Layer | Type | 实测张量 | Cache bytes | Prefill ms | Decode ms |",
            "|---:|:---|:---|---:|---:|---:|",
        ]
    )
    successful = [run for run in result["runs"] if run["status"] == "ok"]
    if successful:
        longest = max(successful, key=lambda item: item["context_length"])
        observations = {item["layer"]: item for item in longest["hook_observations"]}
        for layer, timing in zip(longest["cache_layers"], longest["layer_timings"], strict=True):
            cache_shapes = ", ".join(
                f"{name}={state['shape']} {state['dtype']}" for name, state in layer["states"].items() if state is not None
            )
            observed = observations[layer["layer"]]
            hook_shapes = ", ".join(
                f"{name}={state['shape']} {state['dtype']}"
                for name, state in observed.items()
                if name != "layer" and state is not None
            )
            shapes = "; ".join(part for part in (hook_shapes, cache_shapes) if part)
            lines.append(
                f"| {layer['layer']} | {layer['kind']} | `{shapes}` | {layer['bytes']} | "
                f"{timing['prefill_ms']:.3f} | {timing['decode_ms']:.3f} |"
            )
    else:
        lines.append("| - | - | No successful runtime measurement | - | - | - |")

    lines.extend(["", "## 验证结果", ""])
    verification = result["verification"]
    for name in (
        "cache_roundtrip",
        "attention_cache_injection",
        "gdn_cache_injection",
        "boundary",
        "causal",
        "backend_parity",
    ):
        check = verification.get(name, {})
        lines.append(f"- {name}: **{check.get('status', 'NOT_RUN')}** — {check.get('detail', '')}")

    if verdict.get("reasons"):
        lines.extend(["", "## 未满足的验收条件", ""])
        lines.extend(f"- {reason}" for reason in verdict["reasons"])
    lines.append("")
    return "\n".join(lines)
