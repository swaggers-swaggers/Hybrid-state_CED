"""CUDA runtime imported only by the explicit --run path.

This is a cost surrogate, not a trained early-exit model. Batch size is one;
prefill is full-depth, and all measurements start with generation of token 2.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import torch
from torch import nn
import transformers
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    apply_rotary_pos_emb, create_causal_mask, rotate_half,
)

from .cache import clone_dynamic_cache
from .protocol import ATTENTION_DEPTHS, EXIT_DEPTH, KV_TARGETS, choose_exit, make_plan, scenarios, summarize


class EventProfile:
    """Diagnostic pass only: events include GPU idle gaps, not pure kernel time."""
    def __init__(self):
        self.events = []

    def results(self):
        torch.cuda.synchronize()
        totals = {}
        for label, begin, end in self.events:
            totals[label] = totals.get(label, 0.0) + begin.elapsed_time(end)
        return totals


@contextmanager
def span(profile, label):
    if profile is None:
        yield
        return
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    yield
    end.record()
    profile.events.append((label, begin, end))


def rotate_key(key, cos, sin):
    # Match installed HF partial RoPE without calculating a discarded query.
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    dim = cos.shape[-1]
    rotary, passthrough = key[..., :dim], key[..., dim:]
    return torch.cat((rotary * cos + rotate_half(rotary) * sin, passthrough), dim=-1)


class KVProjector(nn.Module):
    """P_(exit,target)(h_exit) -> (target raw K, target V).

    The sole feature input is the complete exit block's residual output.
    Each pair has independent weights. Position-dependent cache formatting is
    outside this module; no target hidden state, target input norm, historical
    KV or shallower hidden-state fusion is used by the projector.
    """
    def __init__(self, hidden_size, kv_width, *, device, dtype):
        super().__init__()
        self.key = nn.Linear(hidden_size, kv_width, bias=False, device=device, dtype=dtype)
        self.value = nn.Linear(hidden_size, kv_width, bias=False, device=device, dtype=dtype)

    def forward(self, exit_hidden):
        return self.key(exit_hidden), self.value(exit_hidden)


class CostRunner(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.lm = model.model.language_model
        self.config = config
        text = model.config.text_config
        expected = ["full_attention" if d in ATTENTION_DEPTHS else "linear_attention" for d in range(1, 25)]
        if text.num_hidden_layers != 24 or text.hidden_size != 1024 or text.vocab_size != 248320 or text.layer_types != expected:
            raise ValueError("Model architecture differs from the audited Qwen3.5-0.8B layout")
        device, dtype = model.lm_head.weight.device, model.lm_head.weight.dtype
        # One small confidence network at block 12; no vocabulary input.
        self.confidence_head = nn.Sequential(
            nn.Linear(text.hidden_size, config["confidence_hidden_size"], device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(config["confidence_hidden_size"], 1, device=device, dtype=dtype),
        )
        self.readout_map = nn.Linear(text.hidden_size, text.hidden_size, bias=False, device=device, dtype=dtype)
        self.kv_projectors = nn.ModuleDict()
        with torch.no_grad():
            self.readout_map.weight.copy_(torch.eye(text.hidden_size, device=device, dtype=dtype))
            for target in KV_TARGETS:
                attn = self.lm.layers[target - 1].self_attn
                projector = KVProjector(text.hidden_size, attn.k_proj.out_features, device=device, dtype=dtype)
                projector.key.weight.copy_(attn.k_proj.weight)
                projector.value.weight.copy_(attn.v_proj.weight)
                self.kv_projectors[str(target)] = projector
        if len({id(p) for p in self.kv_projectors.values()}) != 3:
            raise AssertionError("Three independent target projectors are required")
        self.requires_grad_(False)
        self.eval()

    def native(self, token, cache):
        hidden = self.lm(input_ids=token, past_key_values=cache, use_cache=True).last_hidden_state
        return self.model.lm_head(hidden[:, -1:, :])

    def project_cache(self, exit_hidden, cache, position_embeddings, profile):
        for target in KV_TARGETS:
            layer = self.lm.layers[target - 1]
            projector = self.kv_projectors[str(target)]
            with span(profile, "kv_projection"):
                raw_key, raw_value = projector(exit_hidden)
            with span(profile, "kv_cache_format_norm_rope"):
                shape = (*exit_hidden.shape[:-1], -1, layer.self_attn.head_dim)
                key = layer.self_attn.k_norm(raw_key.view(shape)).transpose(1, 2)
                value = raw_value.view(shape).transpose(1, 2)
                key = rotate_key(key, *position_embeddings)
            with span(profile, "kv_cache_append"):
                cache.update(key, value, target - 1)
        # Intentionally no read-modify-write of skipped GDN C/S or state flags.

    def step(self, token, cache, case, profile=None):
        with span(profile, "embedding_mask_rope"):
            hidden = self.lm.embed_tokens(token)
            positions = torch.arange(token.shape[1], device=token.device) + cache.get_seq_length()
            positions = positions.view(1, 1, -1).expand(4, token.shape[0], -1)
            text_positions = positions[0]
            causal = create_causal_mask(config=self.model.config.text_config, inputs_embeds=hidden,
                attention_mask=None, past_key_values=cache, position_ids=text_positions)
            linear_mask = self.lm._update_linear_attn_mask(None, cache)
            rope = self.lm.rotary_emb(hidden, positions[1:])
        exited = False
        logits = None
        for index in range(24):
            with span(profile, "executed_blocks"):
                hidden = self.lm.layers[index](hidden, position_embeddings=rope,
                    attention_mask=causal if index + 1 in ATTENTION_DEPTHS else linear_mask,
                    position_ids=text_positions, past_key_values=cache, use_cache=True)
            if index + 1 == EXIT_DEPTH and case["policy"] != "disabled":
                with span(profile, "confidence_network"):
                    confidence_logit = self.confidence_head(hidden)
                with span(profile, "confidence_decision_sync"):
                    score = confidence_logit.float().sigmoid().item()
                    exited = choose_exit(score, case["policy"], self.config["confidence_threshold"])
                if exited:
                    with span(profile, "accepted_readout_matrix"):
                        mapped = self.readout_map(hidden)
                    with span(profile, "accepted_vocabulary"):
                        logits = self.model.lm_head(mapped)
                    self.project_cache(hidden, cache, rope, profile)
                    break
        if not exited:
            with span(profile, "native_final_norm_vocab"):
                logits = self.model.lm_head(self.lm.norm(hidden))
        with span(profile, "greedy_selection"):
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        return next_token, logits, exited


def assert_cache_equal(left, right):
    for a, b in zip(left.layers, right.layers, strict=True):
        for name in ("keys", "values", "conv_states", "recurrent_states"):
            x, y = getattr(a, name, None), getattr(b, name, None)
            if (x is None) != (y is None):
                raise AssertionError(f"Cache presence differs for {name}")
            if x is not None:
                torch.testing.assert_close(x, y, rtol=0, atol=0)
        if getattr(a, "has_previous_state", None) != getattr(b, "has_previous_state", None):
            raise AssertionError("GDN state flags differ")


def validate_runtime(runner, base, first):
    """After --run only: validate both gate branches and the single exit."""
    cases = scenarios()
    config = runner.model.config
    for case in cases[:2]:
        native_cache = clone_dynamic_cache(base, config)
        manual_cache = clone_dynamic_cache(base, config)
        token = first
        for _ in range(2):
            expected = runner.native(token, native_cache)
            actual_token, actual, exited = runner.step(token, manual_cache, case)
            if exited:
                raise AssertionError("Full/rejected path unexpectedly exited")
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-4)
            if not torch.equal(actual_token, expected[:, -1, :].argmax(-1, keepdim=True)):
                raise AssertionError("Native/manual greedy token mismatch")
            assert_cache_equal(native_cache, manual_cache)
            token = actual_token
        del native_cache, manual_cache

    for case in cases:
        cache = clone_dynamic_cache(base, config)
        events, exit_output = [], {}
        def block_hook(module, inputs, output, *, depth):
            events.append(f"block_{depth}")
            if depth == EXIT_DEPTH:
                exit_output["hidden"] = output
        def head_hook(module, inputs, output):
            if inputs[0] is not exit_output["hidden"]:
                raise AssertionError("Confidence head must receive h12 directly")
            events.append("confidence")
        def projection_hook(module, inputs, output, *, target):
            if len(inputs) != 1 or inputs[0] is not exit_output["hidden"]:
                raise AssertionError("Projector must receive only h12")
            if len(output) != 2 or any(tuple(x.shape) != (1, 1, 512) for x in output):
                raise AssertionError("Projector must return target raw K/V [1,1,512]")
            events.append(f"project_{target}")
        handles = [layer.register_forward_hook(lambda m, a, o, d=d: block_hook(m, a, o, depth=d)) for d, layer in enumerate(runner.lm.layers, 1)]
        handles.append(runner.confidence_head.register_forward_hook(head_hook))
        handles.append(runner.readout_map.register_forward_hook(lambda m, a, o: events.append("readout_map")))
        handles.append(runner.model.lm_head.register_forward_hook(lambda m, a, o: events.append("vocabulary")))
        handles += [p.register_forward_hook(lambda m, a, o, target=target: projection_hook(m, a, o, target=target)) for target, p in runner.kv_projectors.items()]
        token = first
        try:
            for step in range(2):
                events.clear()
                token, logits, exited = runner.step(token, cache, case)
                expected_events = [f"block_{d}" for d in range(1, 13)]
                if case["policy"] != "disabled":
                    expected_events += ["confidence"]
                if case["policy"] == "force_accept":
                    expected_events += ["readout_map", "vocabulary", "project_16", "project_20", "project_24"]
                else:
                    expected_events += [f"block_{d}" for d in range(13, 25)] + ["vocabulary"]
                if events != expected_events or exited != (case["policy"] == "force_accept"):
                    raise AssertionError(f"Branch execution order differs: {events}")
                if not bool(torch.isfinite(logits).all()):
                    raise AssertionError("Non-finite logits")
                for d in ATTENTION_DEPTHS:
                    layer = cache.layers[d - 1]
                    if layer.keys.shape[-2] != base.get_seq_length() + step + 1 or layer.values.shape != layer.keys.shape:
                        raise AssertionError("Full-attention KV did not grow by one token")
                    if not bool(torch.isfinite(layer.keys).all() & torch.isfinite(layer.values).all()):
                        raise AssertionError("Non-finite KV cache")
                if exited:
                    for d in range(13, 25):
                        if d in ATTENTION_DEPTHS:
                            continue
                        for name in ("conv_states", "recurrent_states"):
                            torch.testing.assert_close(getattr(base.layers[d - 1], name), getattr(cache.layers[d - 1], name), rtol=0, atol=0)
                        if base.layers[d - 1].has_previous_state != cache.layers[d - 1].has_previous_state:
                            raise AssertionError("Skipped GDN flag changed")
        finally:
            for handle in handles:
                handle.remove()
        before = cache.get_seq_length()
        _, logits, exited = runner.step(token, cache, cases[0])
        if exited or cache.get_seq_length() != before + 1 or not bool(torch.isfinite(logits).all()):
            raise AssertionError("Cannot resume full decode")
        hidden = runner.lm.embed_tokens(first)
        pos = torch.full((3, 1, 1), cache.get_seq_length(), device=first.device, dtype=torch.long)
        rope = runner.lm.rotary_emb(hidden, pos)
        key = cache.layers[3].keys[:, :, -1:, :]
        _, reference_key = apply_rotary_pos_emb(key, key, *rope)
        torch.testing.assert_close(rotate_key(key, *rope), reference_key, rtol=0, atol=0)
        del cache
    return {"status": "PASS", "checks": "native parity for full/reject; exact confidence-before-vocab call order; three direct h12 projectors; KV growth; held deep GDN C/S and flags; full resume; partial RoPE"}


PROMPTS = (
    "研究人员正在分析语言模型的推理速度。实验应记录输入长度、每个生成步骤的耗时，以及模型缓存的变化。请继续解释这个实验。\n",
    "A program receives a sequence of integers. Explain how to compute the running sum, check boundary cases, and report the result.\n",
    "def running_sum(values):\n    total = 0\n    for value in values:\n        total += value\n        yield total\n# Explain the code and give an example.\n",
)


def prompt_ids(tokenizer, length, variant, device):
    # Repeat a natural text fragment to obtain an exact, unpadded token count.
    ids = tokenizer.encode(PROMPTS[variant % len(PROMPTS)], add_special_tokens=False)
    offset = (variant // len(PROMPTS)) % len(ids)
    ids = ids[offset:] + ids[:offset]
    ids = (ids * ((length + len(ids) - 1) // len(ids)))[:length]
    return torch.tensor([ids], dtype=torch.long, device=device)


def decode_loop(runner, base, first, case, count, profiling=False):
    cache = clone_dynamic_cache(base, runner.model.config)
    profile = EventProfile() if profiling else None
    token = first
    generated = []
    exit_count = 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    initial_allocated = torch.cuda.memory_allocated()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start = time.perf_counter()
    begin.record()
    for index in range(count):
        output, logits, exited = runner.step(token, cache, case, profile)
        generated.append(output)
        exit_count += int(exited)
        # Drop the large vocabulary tensor before the next forward in ALL cases.
        del logits
        token = output
    end.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - start) * 1000
    peak = torch.cuda.max_memory_allocated()
    row = {"wall_ms": wall_ms, "wall_ms_per_token": wall_ms / count,
           "gpu_ms_per_token": begin.elapsed_time(end) / count,
           "peak_allocated_mib": peak / 2**20,
           "peak_growth_mib": (peak - initial_allocated) / 2**20,
           "decode_tokens": count, "exit_count": exit_count,
           "generated_token_ids": torch.cat(generated, dim=1).cpu().tolist()[0]}
    if profile is not None:
        row["profile_gpu_span_ms_per_token"] = {k: v / count for k, v in profile.results().items()}
    del cache
    return row


def callable_name(value):
    return None if value is None else f"{getattr(value, '__module__', '')}.{getattr(value, '__name__', type(value).__name__)}"


def write_report(folder, result):
    (folder / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    summary = result["summary"]
    with (folder / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)



@torch.inference_mode()
def run(config, root: Path, output: Path | None):
    plan = make_plan(config)
    if transformers.__version__ != "5.12.1":
        raise RuntimeError("This runner is source-audited for transformers 5.12.1; review the installed API before changing this guard")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; no CPU fallback is allowed")
    folder = output or root / "results" / "exit_cost" / datetime.now().strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    result = {"status": "RUNNING", "plan": plan, "runs": [], "profiles": [], "validation": []}
    try:
        model_path = Path(config["model_path"])
        if not model_path.is_absolute():
            model_path = root / model_path
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        rng = random.Random(config["seed"])
        model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation=config["attention_backend"]).eval().to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        runner = CostRunner(model, config)
        first_linear = next(layer.linear_attn for layer in runner.lm.layers if hasattr(layer, "linear_attn"))
        metadata = model_path / ".cache/huggingface/download/model.safetensors-00001-of-00001.safetensors.metadata"
        result["environment"] = {
            "timestamp": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
            "model_path": str(model_path), "dtype": str(model.lm_head.weight.dtype),
            "attention": model.config.text_config._attn_implementation,
            "gdn_prefill": callable_name(first_linear.chunk_gated_delta_rule),
            "gdn_decode": callable_name(first_linear.recurrent_gated_delta_rule),
            "conv_decode": callable_name(first_linear.causal_conv1d_update),
            "checkpoint_metadata": metadata.read_text().splitlines() if metadata.exists() else None,
            "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in ("runtime.py", "protocol.py", "cache.py")},
        }
        result["weight_policy"] = {"trained": False, "confidence_network": [1024, config["confidence_hidden_size"], 1], "readout": "identity M12, evaluated only after acceptance; shared frozen vocabulary", "projectors": "three independent raw-h12->target-KV maps for 16/20/24, initialized from target native K/V", "interpretation": "Forced controls measure branch costs; optional adaptive uses an untrained network. No calibrated confidence or quality claim."}
        for context in config["context_lengths"]:
            for variant in range(config["prompt_variants"]):
                ids = prompt_ids(tokenizer, context, variant, "cuda")
                base = DynamicCache(config=model.config)
                # A causal prefill's last-position full-model logits produce token 1.
                # No extra dummy decode (which would accidentally exclude token 2).
                first_logits = runner.native(ids, base)
                first = first_logits[:, -1, :].argmax(-1, keepdim=True)
                del first_logits
                result["validation"].append({"context": context, "variant": variant, **validate_runtime(runner, base, first)})
                for case in scenarios(config["include_adaptive"]):
                    decode_loop(runner, base, first, case, config["warmup_tokens"])
                for repeat in range(config["repeats"]):
                    cases = scenarios(config["include_adaptive"])
                    rng.shuffle(cases)  # Paired group, randomized order reduces thermal/order bias.
                    for order, case in enumerate(cases):
                        row = decode_loop(runner, base, first, case, config["decode_tokens"])
                        row.update(context=context, variant=variant, repeat=repeat, order=order, scenario=case["name"])
                        result["runs"].append(row)
                        with (folder / "runs.jsonl").open("a") as stream:
                            stream.write(json.dumps(row) + "\n")
                    print(f"context={context}, prompt={variant + 1}, repeat={repeat + 1}/{config['repeats']}: paired group complete", flush=True)
                for case in scenarios(config["include_adaptive"]):
                    row = decode_loop(runner, base, first, case, config["profile_tokens"], profiling=True)
                    row.update(context=context, variant=variant, scenario=case["name"])
                    result["profiles"].append(row)
                del base
        result["summary"] = summarize(result["runs"])
        result["status"] = "PASS"
        write_report(folder, result)
        print(f"Complete: {folder / 'summary.csv'}", flush=True)
    except BaseException as error:
        result["status"] = "FAILED"
        result["error"] = f"{type(error).__name__}: {error}"
        (folder / "failure.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        raise
