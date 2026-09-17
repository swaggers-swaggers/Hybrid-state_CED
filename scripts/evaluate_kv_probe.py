#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import Qwen3_5ForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1.io import TokenSequenceStore, checkpoint_metadata, load_settings, write_json  # noqa: E402
from phase1.metrics import MetricAccumulator, gate_decision  # noqa: E402
from phase1.models import FullRankKVProbe, build_trainable_probes, parameter_count  # noqa: E402
from phase1.reporting import render_report, save_plot  # noqa: E402
from phase1.teacher import QwenFeatureCapture, attention_core_output, teacher_raw_kv, TARGET_SEMANTICS, require_corrected_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase 1 K/V probes on held-out text")
    parser.add_argument("--config", default="configs/qwen35_08b_phase1.yaml")
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--target-layer", type=int)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def evaluate_split(
    model: Qwen3_5ForConditionalGeneration,
    probes: torch.nn.ModuleDict,
    random_probe: torch.nn.Module,
    settings: dict[str, Any],
    split: str,
    max_sequences: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    dtype = getattr(torch, settings["dtype"])
    source_layers = settings["probe"]["source_layers"]
    target_layer = settings["probe"]["target_layer"]
    attention = model.model.language_model.layers[target_layer].self_attn
    rotary_emb = model.model.language_model.rotary_emb
    store = TokenSequenceStore(settings["data_dir"], split)
    accumulator = MetricAccumulator()
    method_order = ["zero", "random_linear", "original_projection", *list(probes.keys())]
    start = time.perf_counter()
    sequence_count = 0

    probes.eval()
    with QwenFeatureCapture(model, source_layers, target_layer) as capture:
        for input_ids in store.batches(
            settings["probe"]["batch_size"],
            device,
            shuffle=False,
            seed=settings["seed"],
            max_sequences=max_sequences,
        ):
            sources, target_hidden = capture.capture(input_ids)
            target_k, target_v = teacher_raw_kv(attention, target_hidden)
            teacher_attention, teacher_query = attention_core_output(
                attention, rotary_emb, target_hidden, target_k, target_v
            )
            predictions: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
                "zero": (torch.zeros_like(target_k), torch.zeros_like(target_v)),
                "original_projection": (
                    attention.k_proj(model.model.language_model.layers[target_layer].input_layernorm(sources[max(source_layers)])),
                    attention.v_proj(model.model.language_model.layers[target_layer].input_layernorm(sources[max(source_layers)])),
                ),
            }
            with torch.autocast(device_type="cuda", dtype=dtype):
                predictions["random_linear"] = random_probe(sources)
                for name, probe in probes.items():
                    predictions[name] = probe(sources)

            for name in method_order:
                predicted_k, predicted_v = predictions[name]
                predicted_attention, _query = attention_core_output(
                    attention,
                    rotary_emb,
                    target_hidden,
                    predicted_k,
                    predicted_v,
                    teacher_query=teacher_query,
                )
                accumulator.update_tensor(name, "k", predicted_k, target_k, attention.head_dim)
                accumulator.update_tensor(name, "v", predicted_v, target_v, attention.head_dim)
                accumulator.update_tensor(name, "attention_output", predicted_attention, teacher_attention, attention.head_dim)
            sequence_count += input_ids.shape[0]
            if sequence_count % 128 == 0:
                print(f"[phase1:eval] {split}: {sequence_count}/{max_sequences}", flush=True)

    metrics, batch_metrics = accumulator.finalize(method_order)
    return {
        "split": split,
        "num_sequences": sequence_count,
        "num_tokens": sequence_count * store.sequence_length,
        "elapsed_seconds": time.perf_counter() - start,
        "metrics": metrics,
        "batch_metrics": batch_metrics,
        "target_cache_verification": capture.cache_verification,
    }


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    if args.target_layer is not None:
        settings["probe"]["target_layer"] = args.target_layer
    if args.checkpoint_path is not None:
        settings["checkpoint_path"] = str((PROJECT_ROOT / args.checkpoint_path).resolve())
    if args.output_dir is not None:
        settings["output_dir"] = str((PROJECT_ROOT / args.output_dir).resolve())
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 1 evaluation requires CUDA")
    set_seed(settings["seed"])
    device = torch.device(settings["device"])
    dtype = getattr(torch, settings["dtype"])
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        settings["model_path"], dtype=dtype, local_files_only=True
    ).eval().to(device)
    text_config = model.config.text_config
    kv_size = text_config.num_key_value_heads * text_config.head_dim
    probes = build_trainable_probes(
        text_config.hidden_size,
        kv_size,
        settings["probe"]["source_layers"],
        settings["probe"]["low_ranks"],
    ).to(device)
    random_probe = FullRankKVProbe(text_config.hidden_size, kv_size).to(device).eval()
    checkpoint = torch.load(settings["checkpoint_path"], map_location="cpu", weights_only=True)
    require_corrected_checkpoint(checkpoint)
    probes.load_state_dict(checkpoint["probe_state_dict"])
    random_probe.load_state_dict(checkpoint["random_probe_state_dict"])

    max_sequences = args.max_sequences or settings["evaluation"]["max_sequences"]
    split_results = {}
    for split in settings["evaluation"]["splits"]:
        split_results[split] = evaluate_split(model, probes, random_probe, settings, split, max_sequences)

    trained_names = list(probes.keys())
    selected_method = min(
        trained_names,
        key=lambda name: split_results["validation"]["metrics"][name]["attention_output_nmse"],
    )
    gate = gate_decision(
        split_results["test"]["metrics"],
        split_results["test"]["batch_metrics"],
        settings["evaluation"]["bootstrap_samples"],
        settings["gate"]["minimum_point_improvement"],
        settings["gate"]["minimum_ci_improvement"],
        settings["seed"],
        selected_method=selected_method,
    )
    method_order = ["zero", "random_linear", "original_projection", *trained_names]
    parameter_counts = {
        "zero": 0,
        "original_projection": 0,
        "random_linear": parameter_count(random_probe),
        **{name: parameter_count(probe) for name, probe in probes.items()},
    }
    manifest = json.loads((Path(settings["data_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    result = {
        "schema_version": 2,
        "target_semantics": TARGET_SEMANTICS,
        "baseline_semantics": "target input_layernorm(H12) then original k_proj/v_proj",
        "settings": settings,
        "environment": {"model_path": settings["model_path"], **checkpoint_metadata(settings["model_path"]), "torch": torch.__version__},
        "data": manifest,
        "checkpoint": {key: checkpoint[key] for key in ("step", "train_tokens", "elapsed_seconds", "fusion_weights")},
        "evaluation_sequences": max_sequences,
        "method_order": method_order,
        "parameter_counts": parameter_counts,
        "selected_method": selected_method,
        "splits": split_results,
        "gate": gate,
    }
    output_dir = Path(settings["output_dir"])
    write_json(output_dir / "phase1_results.json", result)
    write_json(output_dir / "phase1_batch_metrics.json", {name: value["batch_metrics"] for name, value in split_results.items()})
    (output_dir / "phase1_report.md").write_text(render_report(result), encoding="utf-8")
    save_plot(result, output_dir / "phase1_metrics.png")
    print(json.dumps({"gate": gate, "report": str(output_dir / 'phase1_report.md')}, ensure_ascii=False, indent=2))
    return 0 if gate["status"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
