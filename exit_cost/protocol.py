"""Model-free protocol for a single confidence gate at block 12."""
from __future__ import annotations
import math
import statistics

EXIT_DEPTH = 12
KV_TARGETS = (16, 20, 24)
ATTENTION_DEPTHS = (4, 8, 12, 16, 20, 24)


def scenarios(include_adaptive=False):
    cases = [
        {"name": "full", "policy": "disabled"},
        {"name": "gate_reject", "policy": "force_reject"},
        {"name": "exit_12", "policy": "force_accept"},
    ]
    if include_adaptive:
        cases.append({"name": "adaptive_12", "policy": "network"})
    return cases


def choose_exit(score, policy, threshold):
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Confidence score must be a finite probability")
    if policy == "force_accept":
        return True
    if policy == "force_reject":
        return False
    if policy == "network":
        return score >= threshold
    raise ValueError(f"Unexpected enabled gate policy: {policy}")


def validate(config):
    if config.get("exit_depth") != EXIT_DEPTH or config.get("kv_targets") != list(KV_TARGETS):
        raise ValueError("This experiment requires one exit at 12 and projectors to 16/20/24")
    if config.get("kv_projection") != "exit_hidden_linear":
        raise ValueError("kv_projection must be exit_hidden_linear")
    for field in ("prompt_variants", "repeats", "decode_tokens", "warmup_tokens", "profile_tokens", "confidence_hidden_size"):
        if type(config.get(field)) is not int or config[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    lengths = config.get("context_lengths")
    if not isinstance(lengths, list) or not lengths or any(type(n) is not int or n < 8 for n in lengths):
        raise ValueError("context_lengths must be nonempty integers >= 8")
    if len(lengths) != len(set(lengths)):
        raise ValueError("context_lengths must not contain duplicates")
    threshold = config.get("confidence_threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("confidence_threshold must be a finite value in [0,1]")
    if type(config.get("include_adaptive")) is not bool:
        raise ValueError("include_adaptive must be boolean")
    if config.get("dtype") != "bfloat16" or config.get("attention_backend") != "sdpa":
        raise ValueError("This audited protocol requires bfloat16 and sdpa")
    obsolete = {"head_depths", "top_k", "overlap_threshold", "margin_threshold", "confidence", "input_mode"} & config.keys()
    if obsolete:
        raise ValueError(f"Unsupported configuration fields: {sorted(obsolete)}")


def make_plan(config):
    validate(config)
    groups = len(config["context_lengths"]) * config["prompt_variants"]
    cases = scenarios(config["include_adaptive"])
    timed = groups * config["repeats"] * len(cases) * config["decode_tokens"]
    warmup = groups * len(cases) * config["warmup_tokens"]
    profile = groups * len(cases) * config["profile_tokens"]
    work = timed + warmup + profile
    return {
        "status": "PLAN_ONLY_NO_MODEL_LOADED",
        "start": "Full prefill produces output token 1. Timing starts when consuming token 1 to produce token 2.",
        "end": f"Produce output token {config['decode_tokens'] + 1}; fixed length, no EOS stopping.",
        "scenarios": cases,
        "confidence": {"input": "h12 only; no vocabulary logits", "network": [1024, config["confidence_hidden_size"], 1], "threshold": config["confidence_threshold"], "trained": False},
        "projection": {"input": "h12 [B,1,1024]", "outputs": "target raw K and V, each [B,1,512]", "targets": list(KV_TARGETS), "independent_modules": 3},
        "timed_decode_steps": timed, "warmup_decode_steps": warmup, "profile_decode_steps": profile,
        "planning_assumption_ms_per_step": [15, 40],
        "estimated_minutes": [round(work * .015 / 60 + 1, 1), round(work * .040 / 60 + 3, 1)],
        "estimate_note": "Planning budget only; includes 1–3 min for model load, prefill and runtime validation. No performance measured.",
        "config": config,
    }


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot summarize an empty sample")
    at = (len(ordered) - 1) * q
    lo, hi = math.floor(at), math.ceil(at)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)


def summarize(rows):
    baseline = {(r["context"], r["variant"], r["repeat"]): r for r in rows if r["scenario"] == "full"}
    output = []
    for context in sorted({r["context"] for r in rows}):
        for case in scenarios(include_adaptive=True):
            selected = [r for r in rows if r["context"] == context and r["scenario"] == case["name"]]
            if not selected:
                continue
            values = [r["wall_ms_per_token"] for r in selected]
            ratios = [baseline[(r["context"], r["variant"], r["repeat"])]["wall_ms_per_token"] / r["wall_ms_per_token"] for r in selected]
            output.append({
                "context": context, "scenario": case["name"], "samples": len(selected),
                "mean_wall_ms_per_token": statistics.mean(values),
                "median_wall_ms_per_token": statistics.median(values),
                "p95_run_mean_ms_per_token": percentile(values, 0.95),
                "mean_tokens_per_second": statistics.mean(1000 / v for v in values),
                "mean_gpu_span_ms_per_token": statistics.mean(r["gpu_ms_per_token"] for r in selected),
                "median_paired_speedup": statistics.median(ratios),
                "median_paired_latency_reduction_pct": statistics.median(100 * (1 - 1 / r) for r in ratios),
                "peak_allocated_mib": max(r["peak_allocated_mib"] for r in selected),
                "exit_rate": sum(r["exit_count"] for r in selected) / sum(r["decode_tokens"] for r in selected),
            })
    return output
