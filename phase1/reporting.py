from __future__ import annotations

from pathlib import Path
from typing import Any


def render_report(result: dict[str, Any]) -> str:
    gate = result["gate"]
    checkpoint = result["checkpoint"]
    target_layer = result["settings"]["probe"]["target_layer"]
    lines = [
        "# Qwen3.5-0.8B Phase 1 Attention KV Recoverability",
        "",
        f"**Gate 1：{gate['status']}**",
        "",
        (
            f"验证集选择 `{result['selected_method']}`，其在独立测试集上同时改善 K/V reconstruction 与 teacher-query attention output。"
            if gate["status"] == "GO"
            else f"验证集选择 `{result['selected_method']}`，但独立测试集尚未满足全部 Gate 1 条件。"
        ),
        "",
        "## 实验设置",
        "",
        f"- Teacher：`{result['environment']['model_path']}`，revision `{result['environment']['model_revision']}`。",
        f"- Source：H4/H8/H12（decoder layer 3/7/11 输出）；Target：layer {target_layer} raw pre-RoPE K/V。",
        f"- 训练：{checkpoint['train_tokens']:,} tokens，{checkpoint['step']:,} steps，{checkpoint['elapsed_seconds']:.1f} 秒。",
        f"- 序列长度：{result['data']['sequence_length']}；held-out validation/test 各 {result['evaluation_sequences']:,} sequences。",
        f"- Functional metric：固定真实 teacher layer-{target_layer} Query，对 teacher/predicted K/V 应用原 KNorm 与 RoPE 后计算 causal attention output。",
        "- Target 为 input_layernorm 之后的 self_attn 输入；首批目标经 KNorm/RoPE 后必须与真实缓存逐位相等。",
        "- Original Projection 同样对 H12 应用目标层 input_layernorm，再投影；与旧版基线定义不同。",
        "- 本次复用既有 validation/test split，未重新调参；不是全新独立确认集。Gate 1 不替代真实缓存续写门槛。",
        "",
        "## 独立测试集结果",
        "",
        "| Method | Added params | K NMSE | K cosine | V NMSE | V cosine | Attention NMSE | Attention cosine |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    test_metrics = result["splits"]["test"]["metrics"]
    for method in result["method_order"]:
        metric = test_metrics[method]
        lines.append(
            f"| {method} | {result['parameter_counts'][method]:,} | {metric['k_nmse']:.6f} | "
            f"{metric['k_cosine']:.6f} | {metric['v_nmse']:.6f} | {metric['v_cosine']:.6f} | "
            f"{metric['attention_output_nmse']:.6f} | {metric['attention_output_cosine']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Gate 1",
            "",
            f"- Validation-selected method：`{gate['best_method']}`。",
            f"- 测试集平均 K/V NMSE 相对 Original Projection 改善：{gate['kv_nmse_relative_improvement']:.2%}。",
            f"- 测试集 Attention-output NMSE 相对改善：{gate['attention_nmse_relative_improvement']:.2%}。",
            f"- Attention 改善的 paired bootstrap 95% CI：[{gate['attention_improvement_bootstrap']['ci95_low']:.2%}, {gate['attention_improvement_bootstrap']['ci95_high']:.2%}]。",
            "",
            "## 非对称融合权重",
            "",
            f"- K 对 H4/H8/H12 的权重：{checkpoint['fusion_weights']['k']}。",
            f"- V 对 H4/H8/H12 的权重：{checkpoint['fusion_weights']['v']}。",
            "",
            "## 结论",
            "",
        ]
    )
    if gate["status"] == "GO":
        lines.append(f"Layer {target_layer} 的 K/V 可从浅层表示恢复，且功能误差显著优于 Original Projection 与随机基线。")
    else:
        lines.append("当前 probe 未通过 Gate 1；应先调整训练规模、特征融合或 split，再决定是否扩展到全部 upper attention 层。")
    lines.append("")
    return "\n".join(lines)


def save_plot(result: dict[str, Any], path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    target_layer = result["settings"]["probe"]["target_layer"]
    methods = result["method_order"]
    metrics = result["splits"]["test"]["metrics"]
    x = np.arange(len(methods))
    width = 0.25
    figure, axis = plt.subplots(figsize=(12, 5.8))
    axis.bar(x - width, [metrics[name]["k_nmse"] for name in methods], width, label="K NMSE")
    axis.bar(x, [metrics[name]["v_nmse"] for name in methods], width, label="V NMSE")
    axis.bar(x + width, [metrics[name]["attention_output_nmse"] for name in methods], width, label="Attention output NMSE")
    axis.set_yscale("log")
    axis.set_ylabel("NMSE (log scale)")
    axis.set_title(f"Layer {target_layer} KV Recoverability on Held-out WikiText-103")
    axis.set_xticks(x)
    axis.set_xticklabels(methods, rotation=24, ha="right")
    axis.grid(axis="y", which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
