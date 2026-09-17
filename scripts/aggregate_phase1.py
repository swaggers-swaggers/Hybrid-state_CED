#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1.io import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Phase 1 results across upper attention layers")
    parser.add_argument("--results-dir", default="results/phase1_corrected")
    parser.add_argument("--layers", nargs="+", type=int, default=[15, 19, 23])
    return parser.parse_args()


def render_report(aggregate: dict[str, Any]) -> str:
    lines = [
        "# Qwen3.5-0.8B Phase 1 Upper Attention Recoverability",
        "",
        f"**Phase 1：{aggregate['status']}**",
        "",
        (
            "Layers 15/19/23 均通过 Gate 1；是否进入 GDN 仍取决于真实缓存续写复验。"
            if aggregate["status"] == "GO"
            else "至少一个 upper attention layer 未通过 Gate 1，暂不应进入完整 Hybrid-State CED。"
        ),
        "",
        "## 跨层结果",
        "",
        "| Layer | Validation-selected method | Original Attention NMSE | Selected Attention NMSE | Attention improvement | Bootstrap 95% CI | K/V improvement | Gate |",
        "|---:|:---|---:|---:|---:|:---|---:|:---|",
    ]
    for layer in aggregate["layers"]:
        result = aggregate["results"][str(layer)]
        gate = result["gate"]
        selected = result["selected_method"]
        test = result["splits"]["test"]["metrics"]
        interval = gate["attention_improvement_bootstrap"]
        lines.append(
            f"| {layer} | {selected} | {test['original_projection']['attention_output_nmse']:.6f} | "
            f"{test[selected]['attention_output_nmse']:.6f} | {gate['attention_nmse_relative_improvement']:.2%} | "
            f"[{interval['ci95_low']:.2%}, {interval['ci95_high']:.2%}] | "
            f"{gate['kv_nmse_relative_improvement']:.2%} | {gate['status']} |"
        )
    lines.extend(
        [
            "",
            "## 统一实验口径",
            "",
            "- WikiText-103：每层 4,999,936 train tokens，validation/test 各 262,144 tokens。",
            "- Source：H4/H8/H12；Target：各 attention layer 在 input_layernorm 之后的真实 raw pre-RoPE K/V。",
            "- Original Projection 对 H12 施加目标层 input_layernorm 后再投影；旧版结果与新版基线不能直接作同口径百分比对比。",
            "- 首批训练/评估目标均需与真实缓存逐位相等；本次复用既有数据 split，未使用新的独立确认集。",
            "- Validation 选择方法，test 只用于最终 Gate；functional metric 固定对应层的 teacher Query。",
            "- Gate 要求 K/V 与 Attention-output NMSE 均较 Original Projection 至少改善 10%，且 paired bootstrap 95% CI 下界大于 0。",
            "",
            "## 决策",
            "",
        ]
    )
    if aggregate["status"] == "GO":
        lines.append("修正目标后的三层 probe 均通过 Gate 1。先检查真实缓存联合替换与续写结果，通过后再进入单层 GDN pilot。")
    else:
        failed = [str(layer) for layer in aggregate["layers"] if aggregate["results"][str(layer)]["gate"]["status"] != "GO"]
        lines.append(f"Phase 1 未完成。失败层：{', '.join(failed)}。应先增加训练规模或调整 layer-specific fusion/非线性 projector。")
    lines.append("")
    return "\n".join(lines)


def save_plot(aggregate: dict[str, Any], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = aggregate["layers"]
    original = []
    selected = []
    kv_selected = []
    for layer in layers:
        result = aggregate["results"][str(layer)]
        metrics = result["splits"]["test"]["metrics"]
        method = result["selected_method"]
        original.append(metrics["original_projection"]["attention_output_nmse"])
        selected.append(metrics[method]["attention_output_nmse"])
        kv_selected.append((metrics[method]["k_nmse"] + metrics[method]["v_nmse"]) / 2)
    figure, axis = plt.subplots(figsize=(8.6, 5.2))
    axis.plot(layers, original, marker="o", linewidth=2, label="Original projection attention NMSE")
    axis.plot(layers, selected, marker="o", linewidth=2, label="Selected probe attention NMSE")
    axis.plot(layers, kv_selected, marker="s", linewidth=2, label="Selected probe mean KV NMSE")
    axis.set_xticks(layers)
    axis.set_xlabel("Target attention layer")
    axis.set_ylabel("NMSE")
    axis.set_title("Reconstruction Error vs Layer Depth")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    results_dir = (PROJECT_ROOT / args.results_dir).resolve()
    results: dict[str, Any] = {}
    for layer in args.layers:
        path = results_dir / f"layer{layer}" / "phase1_results.json"
        results[str(layer)] = json.loads(path.read_text(encoding="utf-8"))
    passed = all(results[str(layer)]["gate"]["status"] == "GO" for layer in args.layers)
    aggregate = {
        "schema_version": 1,
        "status": "GO" if passed else "NO-GO",
        "layers": args.layers,
        "results": results,
    }
    write_json(results_dir / "phase1_all_layers.json", aggregate)
    (results_dir / "phase1_all_layers.md").write_text(render_report(aggregate), encoding="utf-8")
    save_plot(aggregate, results_dir / "phase1_layer_depth.png")
    print(json.dumps({"status": aggregate["status"], "report": str(results_dir / 'phase1_all_layers.md')}, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
