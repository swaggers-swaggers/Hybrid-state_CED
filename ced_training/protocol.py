"""Standard-library-only configuration, split audit and calibration policy."""
from __future__ import annotations
import hashlib
import json
import math
import random
from pathlib import Path

SEMANTICS = "block12_residual_to_target_self_attn_raw_kv_v1"
MODULES = ("readout_map", "kv_projectors", "confidence_head")
TARGETS = (16, 20, 24)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(config, profile):
    if config["sequence_length"] != 256 or profile not in config["profiles"]:
        raise ValueError("Requires audited 256-token cache and a named profile")
    if config.get("incremental"):
        spec = config["incremental"]
        if (profile != "pilot" or config.get("distilled_data_path") or config["module_epochs"] != 1
                or spec["chunk_tokens"] != 1_000_000 or type(spec["chunk_index"]) is not int
                or not 0 <= spec["chunk_index"] < 10):
            raise ValueError("Incremental mode requires one of ten 1M-target chunks, one epoch, fixed pilot holdouts")
    for key in ("batch_size", "logit_chunk", "module_epochs", "gate_epochs", "gate_batch_size",
                "eval_every_steps", "scale_sequences", "minimum_accepted", "rollout_prompts", "rollout_tokens"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Invalid positive integer {key}")
    for key in ("module_lr", "gate_lr", "gradient_clip", "temperature"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"Invalid positive number {key}")
    for key in ("ce_weight", "kd_weight", "kv_weight", "weight_decay"):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"Invalid nonnegative weight {key}")
    if min(config["ce_weight"] + config["kd_weight"], config["kv_weight"]) <= 0:
        raise ValueError("Both readout and projection supervision must be enabled")
    if not 0 < config["risk_limit"] < 1 or not config["threshold_grid"]:
        raise ValueError("Invalid risk policy")
    if any(not math.isfinite(t) or not 0 <= t <= 1 for t in config["threshold_grid"]):
        raise ValueError("Invalid threshold")
    if any(type(n) is not int or n < 8 for n in config["rollout_contexts"]):
        raise ValueError("Invalid rollout context")


def split_indices(config, profile, manifest):
    """Disjoint windows with large guard gaps; NOT a document-disjoint claim."""
    validate(config, profile)
    if manifest["sequence_length"] != 256:
        raise ValueError("Cached sequence length differs")
    counts = config["profiles"][profile]
    # Fixed pools across profiles: changing pilot size cannot move a held-out boundary.
    pools = {"modules": ("train", 0, 14000, "module_sequences"),
             "gate": ("train", 16000, 19531, "gate_sequences"),
             "dev": ("validation", 0, 256, "dev_sequences"),
             "calibration": ("validation", 512, 1024, "calibration_sequences"),
             "test": ("test", 0, 1024, "test_sequences")}
    result = {}
    for offset, (name, (split, start, end, count_key)) in enumerate(pools.items()):
        if end > manifest["splits"][split]["num_sequences"]:
            raise ValueError(f"Insufficient data for {name}")
        count = counts[count_key]
        if type(count) is not int or not 1 <= count <= end - start:
            raise ValueError(f"Invalid count for {name}")
        indices = list(range(start, end))
        random.Random(config["seed"] + offset).shuffle(indices)
        result[name] = {"split": split, "indices": indices[:count]}
    return result


def make_plan(config, profile, root):
    data = Path(root) / config["data_path"]
    manifest = json.loads((data / "manifest.json").read_text())
    splits = split_indices(config, profile, manifest)
    # No binary scans or model imports on the plan path.
    plan = {"status": "PLAN_ONLY_NO_TRAINING", "profile": profile, "semantics": SEMANTICS,
            "trainable_parameters": 4325633,
            "data": manifest["dataset"], "data_config": manifest["dataset_config"],
            "split_limit": "Disjoint packed windows, article boundaries unavailable; diagnostic only. Test has prior project exposure.",
            "workload": {k: {"sequences": len(v["indices"]), "input_tokens": len(v["indices"]) * (512 if k == "modules" and config.get("distilled_data_path") else 256),
                                "next_token_targets": len(v["indices"]) * (256 if k == "modules" and config.get("distilled_data_path") else 255)} for k, v in splits.items()},
            "generated_target_note": "Generated-data target counts are maxima; first EOS is included, later padding is excluded. Prompt labels are masked.",
            "epochs": {"modules": config["module_epochs"], "confidence": config["gate_epochs"]},
            "budget_minutes_not_measured": [2, 5] if profile == "smoke" else [15, 30],
            "timing_assumption": "Natural-data smoke measured 138.35 seconds total and 1551.6 supervised tokens/s in the modules stage. Pilot is an extrapolation; teacher data generation is additional and not included.",
            "stages": ["modules", "confidence", "calibrate", "evaluate"], "config": config}
    if config.get("incremental"):
        from .corpus import chunk_spans
        spec = config["incremental"]
        fresh = json.loads((Path(root) / spec["data_path"] / "manifest.json").read_text())
        counts = [255] * (fresh["num_sequences"] - 1)
        counts.append(fresh["effective_next_token_targets"] - sum(counts))
        spans = chunk_spans(counts, spec["chunk_index"] * spec["chunk_tokens"], spec["chunk_tokens"])
        plan["workload"]["modules"] = {"sequences": len(spans), "input_tokens": len(spans) * 256,
                                         "next_token_targets": spec["chunk_tokens"]}
        plan["chunk_index"] = spec["chunk_index"]
        plan["note"] = "Exactly one fresh 1M-target chunk; masked edge windows; no automatic next chunk. End-of-chunk metrics."
    return plan



def calibrate(scores, correct, grid, risk_limit, minimum_accepted):
    if len(scores) != len(correct) or not scores:
        raise ValueError("Empty or mismatched calibration data")
    if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError("Non-finite/out-of-range score")
    if any(v not in (0, 1, False, True) for v in correct):
        raise ValueError("Labels must be binary")
    curve = []
    for threshold in sorted(set(grid)):
        selected = [i for i, score in enumerate(scores) if score >= threshold]
        errors = sum(not correct[i] for i in selected)
        risk = errors / len(selected) if selected else None
        curve.append({"threshold": threshold, "accepted": len(selected), "errors": errors,
                      "coverage": len(selected) / len(scores), "disagreement": risk})
    eligible = [r for r in curve if r["accepted"] >= minimum_accepted and r["disagreement"] <= risk_limit]
    best = max(eligible, key=lambda r: (r["coverage"], r["threshold"])) if eligible else None
    return {"status": "EMPIRICAL_THRESHOLD" if best else "NO_ELIGIBLE_THRESHOLD",
            "threshold": best["threshold"] if best else None, "selected": best, "curve": curve,
            "interpretation": "Teacher top-1 agreement only; correlated tokens and threshold search preclude a formal risk guarantee. No eligible threshold disables early exit."}
