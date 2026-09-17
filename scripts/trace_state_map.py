#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import transformers
import yaml
from transformers import AutoConfig, Qwen3_5ForConditionalGeneration
from transformers.cache_utils import DynamicCache

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase0.cache_tools import (  # noqa: E402
    assert_cache_storage_independent,
    clone_dynamic_cache,
    compare_logits,
    compare_tensors,
    describe_cache,
)
from phase0.reporting import render_markdown, write_json  # noqa: E402
from phase0.state_map import build_static_state_map, tensor_metadata, validate_architecture  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace Qwen3.5 heterogeneous inference state")
    parser.add_argument("--config", default="configs/qwen35_08b_state_map.yaml")
    parser.add_argument("--context-lengths", nargs="+", type=int)
    parser.add_argument("--profile-runs", type=int)
    parser.add_argument("--warmup-runs", type=int)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def load_settings(args: argparse.Namespace) -> dict[str, Any]:
    config_path = (PROJECT_ROOT / args.config).resolve()
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.context_lengths is not None:
        settings["context_lengths"] = args.context_lengths
    if args.profile_runs is not None:
        settings["profile_runs"] = args.profile_runs
    if args.warmup_runs is not None:
        settings["warmup_runs"] = args.warmup_runs
    if args.output_dir is not None:
        settings["output_dir"] = args.output_dir
    settings["config_path"] = str(config_path)
    settings["model_path"] = str((PROJECT_ROOT / settings["model_path"]).resolve())
    settings["output_dir"] = str((PROJECT_ROOT / settings["output_dir"]).resolve())
    return settings


def deterministic_ids(length: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    # Stay away from Qwen multimodal/control token ids while covering varied embeddings.
    ids = ((torch.arange(length, device=device, dtype=torch.long) * 7919) % 200_000) + 1_000
    if int(ids.max()) >= vocab_size:
        ids %= vocab_size
    return ids.unsqueeze(0)


def function_name(function: Any) -> str:
    if function is None:
        return "none"
    return f"{getattr(function, '__module__', type(function).__module__)}.{getattr(function, '__name__', type(function).__name__)}"


def backend_fingerprint(model: Qwen3_5ForConditionalGeneration) -> dict[str, Any]:
    text_model = model.model.language_model
    first_linear = next(layer.linear_attn for layer in text_model.layers if hasattr(layer, "linear_attn"))
    return {
        "attention": model.config.text_config._attn_implementation,
        "gdn_prefill": function_name(first_linear.chunk_gated_delta_rule),
        "gdn_decode": function_name(first_linear.recurrent_gated_delta_rule),
        "causal_conv_prefill": function_name(first_linear.causal_conv1d_fn),
        "causal_conv_decode": function_name(first_linear.causal_conv1d_update),
    }


def nvidia_field(field: str) -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
            check=True,
            text=True,
            capture_output=True,
        )
        return completed.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unavailable"


def checkpoint_metadata(model_path: str) -> dict[str, str]:
    metadata_path = (
        Path(model_path)
        / ".cache"
        / "huggingface"
        / "download"
        / "model.safetensors-00001-of-00001.safetensors.metadata"
    )
    try:
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
        return {"model_revision": lines[0], "weights_etag_sha256": lines[1]}
    except (OSError, IndexError):
        return {"model_revision": "unavailable", "weights_etag_sha256": "unavailable"}


class TraceHooks:
    def __init__(self, model: Qwen3_5ForConditionalGeneration):
        self.model = model
        self.handles: list[Any] = []
        self.starts: dict[int, torch.cuda.Event] = {}
        self.ends: dict[int, torch.cuda.Event] = {}
        self.observations: dict[int, dict[str, Any]] = {
            index: {} for index in range(len(model.model.language_model.layers))
        }

    def _record_tensor(self, index: int, name: str) -> Callable[..., None]:
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(tensor):
                self.observations[index][name] = tensor_metadata(tensor)

        return hook

    def _layer_pre(self, index: int) -> Callable[..., None]:
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            if inputs and torch.is_tensor(inputs[0]):
                self.observations[index]["hidden_input"] = tensor_metadata(inputs[0])
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.starts[index] = event

        return hook

    def _layer_post(self, index: int) -> Callable[..., None]:
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(tensor):
                self.observations[index]["hidden_output"] = tensor_metadata(tensor)
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.ends[index] = event

        return hook

    def __enter__(self) -> "TraceHooks":
        for index, layer in enumerate(self.model.model.language_model.layers):
            self.handles.append(layer.register_forward_pre_hook(self._layer_pre(index)))
            self.handles.append(layer.register_forward_hook(self._layer_post(index)))
            if hasattr(layer, "self_attn"):
                self.handles.append(layer.self_attn.k_proj.register_forward_hook(self._record_tensor(index, "raw_k")))
                self.handles.append(layer.self_attn.k_norm.register_forward_hook(self._record_tensor(index, "knorm_k")))
                self.handles.append(layer.self_attn.v_proj.register_forward_hook(self._record_tensor(index, "raw_v")))
        return self

    def __exit__(self, *_args: Any) -> None:
        for handle in self.handles:
            handle.remove()

    def timings(self) -> dict[int, float]:
        torch.cuda.synchronize()
        return {
            index: self.starts[index].elapsed_time(self.ends[index])
            for index in self.starts.keys() & self.ends.keys()
        }


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "runs": len(values),
    }


@torch.inference_mode()
def one_backbone_pass(
    model: Qwen3_5ForConditionalGeneration,
    input_ids: torch.Tensor,
    cache: DynamicCache,
) -> tuple[float, dict[int, float], dict[int, dict[str, Any]]]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with TraceHooks(model) as hooks:
        start.record()
        outputs = model.model.language_model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        layer_times = hooks.timings()
        observations = hooks.observations
    del outputs
    return elapsed, layer_times, observations


@torch.inference_mode()
def profile_context(
    model: Qwen3_5ForConditionalGeneration,
    length: int,
    settings: dict[str, Any],
) -> dict[str, Any]:
    device = next(model.parameters()).device
    input_ids = deterministic_ids(length, model.config.text_config.vocab_size, device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_vram = torch.cuda.memory_allocated()

    for _ in range(settings["warmup_runs"]):
        warm_cache = DynamicCache(config=model.config)
        one_backbone_pass(model, input_ids, warm_cache)
        del warm_cache
    torch.cuda.synchronize()

    prefill_times: list[float] = []
    prefill_layer_times: dict[int, list[float]] = {index: [] for index in range(model.config.text_config.num_hidden_layers)}
    retained_cache: DynamicCache | None = None
    retained_observations: dict[int, dict[str, Any]] = {}

    for run_index in range(settings["profile_runs"]):
        cache = DynamicCache(config=model.config)
        elapsed, layer_times, observations = one_backbone_pass(model, input_ids, cache)
        prefill_times.append(elapsed)
        for index, value in layer_times.items():
            prefill_layer_times[index].append(value)
        if run_index == settings["profile_runs"] - 1:
            retained_cache = cache
            retained_observations = observations
        else:
            del cache

    assert retained_cache is not None
    cache_layers, cache_bytes = describe_cache(retained_cache)
    decode_id = deterministic_ids(1, model.config.text_config.vocab_size, device)
    decode_times: list[float] = []
    decode_layer_times: dict[int, list[float]] = {index: [] for index in range(model.config.text_config.num_hidden_layers)}

    for _ in range(settings["profile_runs"]):
        decode_cache = clone_dynamic_cache(retained_cache, model.config)
        elapsed, layer_times, _observations = one_backbone_pass(model, decode_id, decode_cache)
        decode_times.append(elapsed)
        for index, value in layer_times.items():
            decode_layer_times[index].append(value)
        del decode_cache

    layer_timings = []
    for index in range(model.config.text_config.num_hidden_layers):
        layer_timings.append(
            {
                "layer": index,
                "prefill_ms": statistics.median(prefill_layer_times[index]),
                "decode_ms": statistics.median(decode_layer_times[index]),
            }
        )

    result = {
        "status": "ok",
        "context_length": length,
        "cache_bytes": cache_bytes,
        "baseline_vram_bytes": baseline_vram,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "prefill_ms": summarize(prefill_times),
        "decode_ms": summarize(decode_times),
        "cache_layers": cache_layers,
        "layer_timings": layer_timings,
        "hook_observations": [
            {"layer": index, **retained_observations[index]}
            for index in range(model.config.text_config.num_hidden_layers)
        ],
    }
    del retained_cache, input_ids, decode_id
    gc.collect()
    torch.cuda.empty_cache()
    return result


def cache_tensor_checks(source: DynamicCache, injected: DynamicCache) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    all_equal = True
    for index, (left, right) in enumerate(zip(source.layers, injected.layers, strict=True)):
        names = ("conv_states", "recurrent_states") if hasattr(left, "conv_states") else ("keys", "values")
        for name in names:
            left_tensor = getattr(left, name, None)
            right_tensor = getattr(right, name, None)
            if left_tensor is None or right_tensor is None:
                equal = left_tensor is None and right_tensor is None
                comparison = {"both_none": equal}
            else:
                equal = torch.equal(left_tensor, right_tensor)
                comparison = compare_tensors(left_tensor, right_tensor)
            all_equal &= equal
            checks.append({"layer": index, "state": name, "exact_equal": equal, **comparison})
    return {"all_exact_equal": all_equal, "states": checks}


@torch.inference_mode()
def verify_cache_roundtrip(
    model: Qwen3_5ForConditionalGeneration,
    length: int,
    tolerance: float,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    prefix = deterministic_ids(length, model.config.text_config.vocab_size, device)
    next_id = deterministic_ids(1, model.config.text_config.vocab_size, device) + 7
    teacher_cache = DynamicCache(config=model.config)
    model(input_ids=prefix, past_key_values=teacher_cache, use_cache=True, logits_to_keep=1)
    injected_cache = clone_dynamic_cache(teacher_cache, model.config)
    assert_cache_storage_independent(teacher_cache, injected_cache)
    state_checks = cache_tensor_checks(teacher_cache, injected_cache)
    reference = model(input_ids=next_id, past_key_values=teacher_cache, use_cache=True, logits_to_keep=1).logits
    injected = model(input_ids=next_id, past_key_values=injected_cache, use_cache=True, logits_to_keep=1).logits
    comparison = compare_logits(reference, injected)
    passed = bool(state_checks["all_exact_equal"] and comparison["top1_equal"] and comparison["max_abs"] <= tolerance)
    return {
        "status": "PASS" if passed else "FAIL",
        "detail": f"max_abs={comparison.get('max_abs')}, top1_equal={comparison.get('top1_equal')}",
        "tolerance": tolerance,
        "logits": comparison,
        "reinjected_states": state_checks,
    }


@torch.inference_mode()
def verify_boundary(
    model: Qwen3_5ForConditionalGeneration,
    length: int,
    max_abs_tolerance: float,
    max_kl: float,
    min_cosine: float,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    ids = deterministic_ids(length, model.config.text_config.vocab_size, device)
    full = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits
    cache = DynamicCache(config=model.config)
    model(input_ids=ids[:, :-1], past_key_values=cache, use_cache=True, logits_to_keep=1)
    boundary = model(input_ids=ids[:, -1:], past_key_values=cache, use_cache=True, logits_to_keep=1).logits
    comparison = compare_logits(full, boundary)
    passed = bool(
        comparison["top1_equal"]
        and comparison["max_abs"] <= max_abs_tolerance
        and comparison["kl_left_to_right"] <= max_kl
        and comparison["cosine"] >= min_cosine
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "detail": f"max_abs={comparison.get('max_abs')}, top1_equal={comparison.get('top1_equal')}",
        "tolerances": {
            "max_abs": max_abs_tolerance,
            "max_kl": max_kl,
            "min_cosine": min_cosine,
        },
        "logits": comparison,
    }


@torch.inference_mode()
def verify_causality(
    model: Qwen3_5ForConditionalGeneration,
    length: int,
    tolerance: float,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    ids_left = deterministic_ids(length, model.config.text_config.vocab_size, device)
    ids_right = ids_left.clone()
    split = length // 2
    ids_right[:, split:] = (ids_right[:, split:] + 97) % model.config.text_config.vocab_size
    left = model.model.language_model(input_ids=ids_left, use_cache=False).last_hidden_state[:, :split]
    right = model.model.language_model(input_ids=ids_right, use_cache=False).last_hidden_state[:, :split]
    comparison = compare_tensors(left, right)
    passed = bool(comparison["max_abs"] <= tolerance)
    return {
        "status": "PASS" if passed else "FAIL",
        "detail": f"unchanged prefix max_abs={comparison.get('max_abs')}",
        "tolerance": tolerance,
        "hidden": comparison,
    }


def evaluate_verdict(
    settings: dict[str, Any],
    architecture_errors: list[str],
    runs: list[dict[str, Any]],
    verification: dict[str, Any],
) -> dict[str, Any]:
    reasons = list(architecture_errors)
    successful_lengths = {run["context_length"] for run in runs if run["status"] == "ok"}
    missing = sorted(set(settings["context_lengths"]) - successful_lengths)
    if missing:
        reasons.append(f"Missing successful runtime measurements for context lengths: {missing}")
    for name in (
        "cache_roundtrip",
        "attention_cache_injection",
        "gdn_cache_injection",
        "boundary",
        "causal",
        "backend_parity",
    ):
        if verification.get(name, {}).get("status") != "PASS":
            reasons.append(f"Verification {name} did not pass")
    if reasons:
        return {"status": "NO-GO", "summary": "Phase 0 acceptance criteria are not yet all satisfied.", "reasons": reasons}
    return {"status": "PASS", "summary": "The complete inference-state map and cache reinjection gates passed.", "reasons": []}


def main() -> int:
    args = parse_args()
    settings = load_settings(args)
    if settings["device"] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Phase 0 runtime measurements require a visible CUDA GPU")

    torch.manual_seed(settings["seed"])
    torch.cuda.manual_seed_all(settings["seed"])
    dtype = getattr(torch, settings["dtype"])
    official_config = AutoConfig.from_pretrained(settings["model_path"], local_files_only=True)
    static_map = build_static_state_map(official_config.text_config, dtype_name=settings["dtype"])
    architecture_errors = validate_architecture(static_map, settings["expected"])

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        settings["model_path"],
        dtype=dtype,
        local_files_only=True,
    ).eval().to("cuda")
    fingerprint = backend_fingerprint(model)
    environment = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "model_path": settings["model_path"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": str(next(model.parameters()).device),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "driver": nvidia_field("driver_version"),
        "attention_backend": fingerprint["attention"],
        "backend_fingerprint": fingerprint,
        "model_vram_bytes": torch.cuda.memory_allocated(),
        **checkpoint_metadata(settings["model_path"]),
    }

    runs: list[dict[str, Any]] = []
    for length in settings["context_lengths"]:
        print(f"[phase0] profiling context={length}", flush=True)
        try:
            runs.append(profile_context(model, length, settings))
        except torch.OutOfMemoryError as error:
            runs.append({"status": "oom", "context_length": length, "error": str(error)})
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as error:  # Preserve evidence for a partially completed experiment.
            runs.append({"status": "error", "context_length": length, "error": f"{type(error).__name__}: {error}"})
            gc.collect()
            torch.cuda.empty_cache()

    tolerances = settings["tolerances"]
    verification: dict[str, Any] = {}
    try:
        verification["cache_roundtrip"] = verify_cache_roundtrip(
            model, settings["roundtrip_length"], tolerances["roundtrip_max_abs"]
        )
        roundtrip = verification["cache_roundtrip"]
        attention_states = [
            item for item in roundtrip.get("reinjected_states", {}).get("states", []) if item["state"] in {"keys", "values"}
        ]
        gdn_states = [
            item
            for item in roundtrip.get("reinjected_states", {}).get("states", [])
            if item["state"] in {"conv_states", "recurrent_states"}
        ]
        attention_passed = roundtrip["status"] == "PASS" and bool(attention_states) and all(
            item["exact_equal"] for item in attention_states
        )
        gdn_passed = roundtrip["status"] == "PASS" and bool(gdn_states) and all(item["exact_equal"] for item in gdn_states)
        verification["attention_cache_injection"] = {
            "status": "PASS" if attention_passed else "FAIL",
            "detail": f"{len(attention_states)} K/V tensors copied exactly; functional continuation={roundtrip['status']}",
        }
        verification["gdn_cache_injection"] = {
            "status": "PASS" if gdn_passed else "FAIL",
            "detail": f"{len(gdn_states)} recurrent/conv tensors copied exactly; functional continuation={roundtrip['status']}",
        }
    except Exception as error:
        verification["cache_roundtrip"] = {"status": "ERROR", "detail": f"{type(error).__name__}: {error}"}
        verification["attention_cache_injection"] = {"status": "ERROR", "detail": "cache round-trip did not run"}
        verification["gdn_cache_injection"] = {"status": "ERROR", "detail": "cache round-trip did not run"}
    try:
        verification["boundary"] = verify_boundary(
            model,
            settings["boundary_length"],
            tolerances["boundary_max_abs"],
            tolerances["boundary_max_kl"],
            tolerances["boundary_min_cosine"],
        )
    except Exception as error:
        verification["boundary"] = {"status": "ERROR", "detail": f"{type(error).__name__}: {error}"}
    try:
        verification["causal"] = verify_causality(model, settings["causal_length"], tolerances["causal_max_abs"])
    except Exception as error:
        verification["causal"] = {"status": "ERROR", "detail": f"{type(error).__name__}: {error}"}

    fingerprint_after = backend_fingerprint(model)
    parity = fingerprint == fingerprint_after
    verification["backend_parity"] = {
        "status": "PASS" if parity else "FAIL",
        "detail": "prefill/decode functions remained unchanged" if parity else "backend fingerprint changed during run",
        "before": fingerprint,
        "after": fingerprint_after,
    }

    verdict = evaluate_verdict(settings, architecture_errors, runs, verification)
    result = {
        "schema_version": 1,
        "settings": settings,
        "environment": environment,
        "static_state_map": static_map,
        "architecture_errors": architecture_errors,
        "runs": runs,
        "verification": verification,
        "verdict": verdict,
    }
    output_dir = Path(settings["output_dir"])
    write_json(output_dir / "state_map.json", result)
    write_json(output_dir / "verification.json", verification)
    (output_dir / "state_map.md").write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    return 0 if verdict["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
