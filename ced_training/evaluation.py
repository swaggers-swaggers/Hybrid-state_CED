"""Held-out diagnostics: same-prefix continuations, state controls and free generation."""
from __future__ import annotations
import math
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, DynamicCache
from exit_cost.cache import clone_dynamic_cache
from exit_cost.runtime import assert_cache_equal, validate_runtime
from .engine import autocast, evaluate_static, gate_features, gate_metrics, write_json, modules
from .protocol import TARGETS


def distribution_metrics(student, teacher, label):
    p = F.log_softmax(teacher.float().reshape(1, -1), dim=-1)
    q = F.log_softmax(student.float().reshape(1, -1), dim=-1)
    kl = F.kl_div(q, p, reduction="batchmean", log_target=True).item()
    nll = F.cross_entropy(student.float().reshape(1, -1), label.reshape(1)).item()
    teacher_nll = F.cross_entropy(teacher.float().reshape(1, -1), label.reshape(1)).item()
    agreement = int(student.argmax(-1).item() == teacher.argmax(-1).item())
    return kl, nll, teacher_nll, agreement


def correct_state(runner, cache, shadow, kv_oracle, update_gdn):
    """Diagnostic only. Shadow executes the complete step from this branch's own history."""
    if kv_oracle:
        for depth in TARGETS:
            target, source = cache.layers[depth - 1], shadow.layers[depth - 1]
            target.keys[:, :, -1:, :].copy_(source.keys[:, :, -1:, :])
            target.values[:, :, -1:, :].copy_(source.values[:, :, -1:, :])
    if update_gdn:
        for depth in range(13, 25):
            if depth in TARGETS:
                continue
            target, source = cache.layers[depth - 1], shadow.layers[depth - 1]
            # Copy complete states. update_conv_state would shift the cache a second time.
            target.conv_states.copy_(source.conv_states)
            target.recurrent_states.copy_(source.recurrent_states)
            target.has_previous_state = source.has_previous_state
    if kv_oracle and update_gdn:
        assert_cache_equal(cache, shadow)


def same_prefix(runner, base, tokens, context, count, case_name, gate_enabled):
    cache = clone_dynamic_cache(base, runner.model.config)
    reference_cache = clone_dynamic_cache(base, runner.model.config)
    totals = {"kl": 0., "nll": 0., "teacher_nll": 0., "agreements": 0,
              "exit_disagreements": 0, "exits": 0, "resume_kl": 0., "resume_steps": 0}
    for index in range(count):
        token = tokens[:, context + index:context + index + 1]
        label = tokens[:, context + index + 1]
        teacher_logits = runner.native(token, reference_cache)
        if case_name == "adaptive":
            policy = "network" if gate_enabled else "force_reject"
            exiting = False  # actual decision returned by runner
        else:
            # Four successive exits, then one complete step: expose stale memory on resume.
            exiting = index % 5 != 4
            policy = "force_accept" if exiting else "disabled"
        shadow = None
        if exiting and case_name != "projected_kv_held_gdn":
            shadow = clone_dynamic_cache(cache, runner.model.config)
            runner.native(token, shadow)
        _, student_logits, exited = runner.step(token, cache, {"policy": policy})
        if shadow is not None:
            correct_state(runner, cache, shadow, kv_oracle=case_name.startswith("oracle_kv"),
                          update_gdn=case_name.endswith("updated_gdn"))
            del shadow
        kl, nll, tnll, agreement = distribution_metrics(student_logits, teacher_logits, label)
        totals["kl"] += kl
        totals["nll"] += nll
        totals["teacher_nll"] += tnll
        totals["agreements"] += agreement
        totals["exits"] += int(exited)
        totals["exit_disagreements"] += int(exited and not agreement)
        if not exited:
            totals["resume_kl"] += kl
            totals["resume_steps"] += 1
    return {"case": case_name, "steps": count, "mean_kl": totals["kl"] / count,
            "nll": totals["nll"] / count, "teacher_nll": totals["teacher_nll"] / count,
            "ppl_ratio": math.exp(min(80, (totals["nll"] - totals["teacher_nll"]) / count)),
            "top1_agreement": totals["agreements"] / count, "exit_rate": totals["exits"] / count,
            "exit_disagreement": totals["exit_disagreements"] / totals["exits"] if totals["exits"] else None,
            "resume_mean_kl": totals["resume_kl"] / totals["resume_steps"] if totals["resume_steps"] else None}


def greedy(runner, base, first, count, policy):
    cache = clone_dynamic_cache(base, runner.model.config)
    token, outputs, exits = first, [], 0
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(count):
        token, logits, exited = runner.step(token, cache, {"policy": policy})
        outputs.append(token)
        exits += int(exited)
        del logits
    torch.cuda.synchronize()
    seconds = time.perf_counter() - begin
    return {"ids": torch.cat(outputs, dim=-1).squeeze(0).tolist(), "exits": exits,
            "seconds": seconds, "ms_per_token": seconds * 1000 / count}


@torch.inference_mode()
def evaluate(runner, teacher, data, config, output, previous):
    runner.requires_grad_(False)
    for module in modules(runner).values():
        module.to(dtype=torch.bfloat16)
    calibration = previous["extra"]["calibration"]
    threshold = calibration["threshold"]
    gate_enabled = threshold is not None
    if gate_enabled:
        runner.config["confidence_threshold"] = threshold
    static = evaluate_static(runner, teacher, data, config, previous["scales"], split="test")
    features = gate_features(runner, teacher, data, config, "test")
    gate_summary, scores = gate_metrics(runner, features, config)
    selected = [i for i, score in enumerate(scores) if gate_enabled and score >= threshold]
    gate_summary.update({"threshold": threshold, "accepted": len(selected), "coverage": len(selected) / len(scores),
                         "disagreement": sum(not features[1][i].item() for i in selected) / len(selected) if selected else None})
    del features
    tokenizer = AutoTokenizer.from_pretrained(data.path.parents[1] / config["model_path"], local_files_only=True)
    flat = data.arrays["test"].reshape(-1)
    rows, generations, validation = [], [], None
    cases = ("oracle_kv_updated_gdn", "projected_kv_updated_gdn", "oracle_kv_held_gdn", "projected_kv_held_gdn", "adaptive")
    offset = 0
    for context in config["rollout_contexts"]:
        for prompt_index in range(config["rollout_prompts"]):
            length = context + config["rollout_tokens"] + 1
            if offset + length > len(flat):
                raise ValueError("Not enough held-out tokens for disjoint rollout windows")
            tokens = torch.from_numpy(np.array(flat[offset:offset + length], dtype=np.int64)).view(1, -1).cuda()
            offset += length + 256
            base = DynamicCache(config=runner.model.config)
            with autocast():
                first = runner.native(tokens[:, :context], base).argmax(-1)
                if validation is None:
                    validation = validate_runtime(runner, base, first)
                for case in cases:
                    row = same_prefix(runner, base, tokens, context, config["rollout_tokens"], case, gate_enabled)
                    row.update({"context": context, "prompt": prompt_index})
                    rows.append(row)
                policies = [("full", "disabled"), ("adaptive", "network" if gate_enabled else "force_reject")]
                random.Random(config["seed"] + context + prompt_index).shuffle(policies)
                pair = {}
                for name, policy in policies:
                    greedy(runner, base, first, 2, policy)  # warmup on independent cache
                    pair[name] = greedy(runner, base, first, config["rollout_tokens"], policy)
                    pair[name]["text"] = tokenizer.decode([first.item()] + pair[name]["ids"])
                generations.append({"context": context, "prompt": prompt_index,
                                    "prompt_text": tokenizer.decode(tokens[0, :context].tolist()),
                                    "full": pair["full"], "adaptive": pair["adaptive"],
                                    "speedup": pair["full"]["seconds"] / pair["adaptive"]["seconds"]})
    write_json(output / "continuations.json", rows)
    write_json(output / "generations.json", generations)
    write_json(output / "static_test.json", {"modules": static, "confidence": gate_summary})
    return {"status": "EVALUATION_FINISHED", "gate_enabled": gate_enabled,
            "runtime_validation": validation, "static_test": static, "confidence_test": gate_summary,
            "rollout_cases": len(rows), "generation_pairs": len(generations),
            "limits": "Diagnostic English WikiText only. Fixed oracle controls cost full shadow steps and are never speed results. Quality/speed acceptance requires reviewing continuations and generation; no automatic PASS."}
