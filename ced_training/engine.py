"""Staged training; imported only after explicit --run. All data/model access is local."""
from __future__ import annotations
import json
import math
from pathlib import Path
import random
import time
import traceback
from datetime import datetime
import numpy as np
import torch
from torch import nn
from transformers import Qwen3_5ForConditionalGeneration, __version__ as transformers_version
from exit_cost.runtime import CostRunner
from .protocol import SEMANTICS, MODULES, TARGETS, sha256, split_indices, calibrate
from .teacher import Teacher, prediction_rows
from .losses import readout_losses, normalized_mse, gate_loss


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


class Data:
    def __init__(self, root, config, profile):
        self.path = root / config["data_path"]
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        self.manifest_hash = sha256(self.path / "manifest.json")
        self.splits = split_indices(config, profile, self.manifest)
        self.distilled = None
        self.distilled_manifest_hash = None
        self.arrays = {}
        for name, info in self.manifest["splits"].items():
            path = self.path / info["file"]
            if path.stat().st_size != info["num_tokens"] * 4 or sha256(path) != info["sha256"]:
                raise ValueError(f"Corrupt data: {name}")
            self.arrays[name] = np.memmap(path, dtype="<i4", mode="r").reshape(-1, 256)
            if self.arrays[name].min() < 0 or self.arrays[name].max() >= 248320:
                raise ValueError(f"Invalid token IDs: {name}")

        if config.get("distilled_data_path"):
            directory = root / config["distilled_data_path"]
            manifest_path = directory / "manifest.json"
            self.distilled = json.loads(manifest_path.read_text())
            self.distilled_manifest_hash = sha256(manifest_path)
            m = self.distilled
            if (m["status"] != "COMPLETE" or m["source_manifest_sha256"] != self.manifest_hash
                    or m["prompt_indices"] != self.splits["modules"]["indices"]
                    or m["model_weights_sha256"] != self.manifest["weights_etag_sha256"]
                    or m["prompt_length"] != 256 or m["max_new_tokens"] != 256
                    or m["profile"] != profile or m["strategy"] != "greedy_stop_at_eos"):
                raise ValueError("Generated data provenance differs from this training trial")
            path = directory / m["file"]
            if sha256(path) != m["sha256"] or path.stat().st_size != len(m["prompt_indices"]) * 512 * 4:
                raise ValueError("Generated data checksum/shape differs")
            self.generated_array = np.memmap(path, dtype="<i4", mode="r").reshape(-1, 512)
            if self.generated_array.min() < 0 or self.generated_array.max() >= 248320:
                raise ValueError("Invalid generated token IDs")
            source_prompts = self.arrays["train"][m["prompt_indices"]]
            if not np.array_equal(self.generated_array[:, :256], source_prompts):
                raise ValueError("Generated prompts are not the declared training windows")

    def capture(self, teacher, batch, split, verify=False):
        tokens = batch[0] if isinstance(batch, tuple) else batch
        return self.rows(teacher.capture(tokens, verify=verify), batch, split)

    def rows(self, captured, tokens, split):
        if split == "modules" and self.distilled:
            return prediction_rows(captured, tokens, start_position=255,
                                   eos_token_ids=self.distilled["eos_token_ids"])
        return prediction_rows(captured, tokens)

    def batches(self, name, batch_size, *, epoch=None, limit=None):
        spec = self.splits[name]
        generated = name == "modules" and self.distilled is not None
        rows = list(range(len(spec["indices"]))) if generated else list(spec["indices"])
        if limit is not None:
            rows = rows[:limit]
        if epoch is not None:
            random.Random(87231 + epoch).shuffle(rows)
        for i in range(0, len(rows), batch_size):
            source_array = self.generated_array if generated else self.arrays[spec["split"]]
            array = np.array(source_array[rows[i:i + batch_size]], dtype=np.int64)
            yield torch.from_numpy(array).cuda()


def modules(runner):
    return {name: getattr(runner, name) for name in MODULES}


def set_trainable(runner, names):
    runner.requires_grad_(False)
    for name in names:
        getattr(runner, name).requires_grad_(True)
    if any(p.requires_grad for p in runner.model.parameters()):
        raise AssertionError("Backbone/vocabulary must remain frozen")
    return [p for name in names for p in getattr(runner, name).parameters()]


def load_runner(root, config):
    if transformers_version != "5.12.1":
        raise RuntimeError(f"Re-audit required for Transformers {transformers_version}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This experiment requires CUDA with BF16 support")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        root / config["model_path"], local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    runner = CostRunner(model, {"confidence_hidden_size": 128, "confidence_threshold": 0.9})
    # FP32 master parameters, BF16 matmuls under autocast; vocabulary stays frozen BF16.
    for module in modules(runner).values():
        module.float()
    runner.requires_grad_(False)
    return runner


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16)


def checkpoint(path, runner, config, profile, data, stage, scales, extra=None):
    payload = {"schema_version": 1, "semantics": SEMANTICS, "stage": stage,
               "model_revision": data.manifest["model_revision"],
               "model_weights_sha256": data.manifest["weights_etag_sha256"],
               "data_manifest_sha256": data.manifest_hash, "distilled_manifest_sha256": data.distilled_manifest_hash,
               "incremental_manifest_sha256": getattr(data, "incremental_manifest_hash", None),
               "chunk": getattr(data, "chunk", None),
               "config": config, "profile": profile,
               "split_indices": data.splits, "scales": scales, "extra": extra or {},
               "modules": {name: {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
                           for name, module in modules(runner).items()}}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, runner, config, profile, data, expected):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["schema_version"] != 1 or state["semantics"] != SEMANTICS or state["stage"] != expected:
        raise ValueError(f"Expected {expected} checkpoint with current target semantics")
    if (state["config"] != config or state["profile"] != profile or state["split_indices"] != data.splits
            or state["data_manifest_sha256"] != data.manifest_hash
            or state.get("distilled_manifest_sha256") != data.distilled_manifest_hash
            or state.get("incremental_manifest_sha256") != getattr(data, "incremental_manifest_hash", None)
            or state.get("chunk") != getattr(data, "chunk", None)
            or state["model_revision"] != data.manifest["model_revision"]
            or state["model_weights_sha256"] != data.manifest["weights_etag_sha256"]):
        raise ValueError("Checkpoint model/data/config/profile differs; start a separately identified trial")
    if set(state["modules"]) != set(MODULES):
        raise ValueError("Checkpoint must contain exactly the auxiliary modules")
    for name, module in modules(runner).items():
        module.load_state_dict(state["modules"][name], strict=True)
    return state


def load_warm_start(path, runner, config, profile, data):
    if config.get("incremental"):
        from .incremental import load_chunk_warm_start
        return load_chunk_warm_start(path, runner, config, profile, data)
    if not data.distilled:
        raise ValueError("Warm start requires generated data or an incremental chunk")
    source_config = {k: v for k, v in config.items() if k != "distilled_data_path"}
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (state["schema_version"] != 1 or state["semantics"] != SEMANTICS or state["stage"] != "modules"
            or state["config"] != source_config or state["profile"] != profile
            or state["data_manifest_sha256"] != data.manifest_hash or state.get("distilled_manifest_sha256") is not None
            or state["split_indices"] != data.splits or state["model_revision"] != data.manifest["model_revision"]
            or state["model_weights_sha256"] != data.manifest["weights_etag_sha256"]):
        raise ValueError("Warm start must be the matching natural-data pilot modules checkpoint")
    for name in ("readout_map", "kv_projectors"):
        getattr(runner, name).load_state_dict(state["modules"][name], strict=True)
    # Confidence remains freshly initialized; optimizer state is not reused.
    return state


@torch.no_grad()
def target_scales(teacher, data, config):
    sums = {f"{kind}{depth}": 0.0 for depth in TARGETS for kind in ("k", "v")}
    counts = dict.fromkeys(sums, 0)
    for batch_index, tokens in enumerate(data.batches("modules", config["batch_size"], limit=config["scale_sequences"])):
        values, _ = data.capture(teacher, tokens, "modules", verify=batch_index == 0)
        for name in sums:
            sums[name] += values[name].float().square().sum().item()
            counts[name] += values[name].numel()
    return {name: max(sums[name] / counts[name], 1e-8) for name in sums}


def projection_loss(runner, values, scales):
    errors = {}
    for depth in TARGETS:
        k, v = runner.kv_projectors[str(depth)](values["h12"])
        errors[f"k{depth}"] = normalized_mse(k, values[f"k{depth}"], scales[f"k{depth}"])
        errors[f"v{depth}"] = normalized_mse(v, values[f"v{depth}"], scales[f"v{depth}"])
    return torch.stack(list(errors.values())).mean(), errors


@torch.no_grad()
def evaluate_static(runner, teacher, data, config, scales, split="dev"):
    totals, n = {}, 0
    for batch_index, tokens in enumerate(data.batches(split, config["batch_size"])):
        values, labels = data.capture(teacher, tokens, split, verify=batch_index == 0)
        with autocast():
            _, kv = projection_loss(runner, values, scales)
        for name, loss in kv.items():
            totals[name] = totals.get(name, 0.0) + loss.item() * labels.numel()
        for start in range(0, len(labels), config["logit_chunk"]):
            end = min(start + config["logit_chunk"], len(labels))
            with autocast():
                teacher_logits = runner.model.lm_head(values["final"][start:end])
                student_logits = runner.model.lm_head(runner.readout_map(values["h12"][start:end]))
                ce, kd = readout_losses(student_logits, teacher_logits, labels[start:end], config["temperature"])
            teacher_ce = nn.functional.cross_entropy(teacher_logits.float(), labels[start:end])
            agree = (teacher_logits.argmax(-1) == student_logits.argmax(-1)).float().mean()
            for name, value in (("ce", ce), ("kd", kd), ("teacher_ce", teacher_ce), ("top1_agreement", agree)):
                totals[name] = totals.get(name, 0.0) + value.item() * (end - start)
            n += end - start
    metrics = {name: value / n for name, value in totals.items()}
    metrics["kv"] = sum(metrics[f"{kind}{depth}"] for depth in TARGETS for kind in ("k", "v")) / 6
    metrics["selection_loss"] = config["ce_weight"] * metrics["ce"] + config["kd_weight"] * metrics["kd"] + config["kv_weight"] * metrics["kv"]
    metrics["targets"] = n
    return metrics


def train_modules(runner, teacher, data, config, profile, output, warm_state=None):
    params = set_trainable(runner, ("readout_map", "kv_projectors"))
    optimizer = torch.optim.AdamW(params, lr=config["module_lr"], weight_decay=config["weight_decay"])
    scales = warm_state["scales"] if warm_state else target_scales(teacher, data, config)
    best, step, supervised_tokens = math.inf, 0, 0
    initial = evaluate_static(runner, teacher, data, config, scales)
    write_json(output / "initial_dev.json", initial)
    # Include initialization in best selection; regression never silently replaces it.
    best = initial["selection_loss"]
    checkpoint(output / "modules.pt", runner, config, profile, data, "modules", scales, {"step": 0, "dev": initial})
    total_steps = math.ceil(len(data.splits["modules"]["indices"]) / config["batch_size"]) * config["module_epochs"]
    for epoch in range(config["module_epochs"]):
        for tokens in data.batches("modules", config["batch_size"], epoch=epoch):
            values, labels = data.capture(teacher, tokens, "modules", verify=step == 0)
            supervised_tokens += len(labels)
            optimizer.zero_grad(set_to_none=True)
            # Short warmup followed by a cosine decay with a 10% LR floor.
            warmup = max(1, int(total_steps * .03))
            factor = min(1., (step + 1) / warmup) * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / total_steps)))
            optimizer.param_groups[0]["lr"] = config["module_lr"] * factor
            with autocast():
                kv, _ = projection_loss(runner, values, scales)
            (config["kv_weight"] * kv).backward()
            batch_ce, batch_kd = 0., 0.
            for start in range(0, len(labels), config["logit_chunk"]):
                end = min(start + config["logit_chunk"], len(labels))
                fraction = (end - start) / len(labels)
                with autocast():
                    with torch.no_grad():
                        teacher_logits = runner.model.lm_head(values["final"][start:end])
                    student_logits = runner.model.lm_head(runner.readout_map(values["h12"][start:end]))
                    ce, kd = readout_losses(student_logits, teacher_logits, labels[start:end], config["temperature"])
                ((config["ce_weight"] * ce + config["kd_weight"] * kd) * fraction).backward()
                batch_ce += ce.item() * fraction
                batch_kd += kd.item() * fraction
            torch.nn.utils.clip_grad_norm_(params, config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            step += 1
            row = {"step": step, "epoch": epoch + 1, "ce": batch_ce, "kd": batch_kd, "kv": kv.item()}
            if step % config["eval_every_steps"] == 0 or step == total_steps:
                row["dev"] = evaluate_static(runner, teacher, data, config, scales)
                if row["dev"]["selection_loss"] < best:
                    best = row["dev"]["selection_loss"]
                    checkpoint(output / "modules.pt", runner, config, profile, data, "modules", scales, {"step": step, "dev": row["dev"]})
            with (output / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    if config.get("incremental") and supervised_tokens != config["incremental"]["chunk_tokens"]:
        raise AssertionError("Incremental training did not consume exactly one authorized chunk")
    if data.distilled and supervised_tokens != data.distilled["generated_tokens"] * config["module_epochs"]:
        raise AssertionError("Completion loss-mask count differs from the generation manifest")
    checkpoint(output / "last_modules.pt", runner, config, profile, data, "modules", scales, {"step": step})
    return {"status": "MODULES_TRAINED", "steps": step, "best_dev_selection_loss": best,
            "checkpoint": str(output / "modules.pt"), "supervised_tokens": supervised_tokens,
            "training_data": "fresh_natural_chunk" if config.get("incremental") else ("teacher_generated_completions" if data.distilled else "natural_text"),
            "chunk": getattr(data, "chunk", None),
            "warm_started": warm_state is not None}


@torch.no_grad()
def gate_features(runner, teacher, data, config, split):
    hidden, correct = [], []
    for batch_index, tokens in enumerate(data.batches(split, config["batch_size"])):
        values, labels = data.capture(teacher, tokens, split, verify=batch_index == 0)
        hidden.append(values["h12"].cpu())
        for start in range(0, len(labels), config["logit_chunk"]):
            end = min(start + config["logit_chunk"], len(labels))
            with autocast():
                reference = runner.model.lm_head(values["final"][start:end]).argmax(-1)
                candidate = runner.model.lm_head(runner.readout_map(values["h12"][start:end])).argmax(-1)
            correct.append((reference == candidate).float().cpu())
    return torch.cat(hidden), torch.cat(correct)


@torch.no_grad()
def gate_metrics(runner, features, config):
    h, y = features
    scores, total_loss = [], 0.
    for start in range(0, len(y), config["gate_batch_size"]):
        end = start + config["gate_batch_size"]
        with autocast():
            logits = runner.confidence_head(h[start:end].cuda()).squeeze(-1)
            loss = gate_loss(logits, y[start:end].cuda())
        total_loss += loss.item() * len(logits)
        scores.extend(logits.float().sigmoid().cpu().tolist())
    brier = sum((p - label)**2 for p, label in zip(scores, y.tolist())) / len(y)
    return {"bce": total_loss / len(y), "brier": brier, "positive_rate": y.mean().item(), "tokens": len(y)}, scores


def train_confidence(runner, teacher, data, config, profile, output, previous):
    params = set_trainable(runner, ("confidence_head",))
    train = gate_features(runner, teacher, data, config, "gate")
    dev = gate_features(runner, teacher, data, config, "dev")
    # CPU h12 cache is bounded (~255 MiB in pilot), transient and never full logits.
    optimizer = torch.optim.AdamW(params, lr=config["gate_lr"], weight_decay=config["weight_decay"])
    best, history = math.inf, []
    h, y = train
    for epoch in range(config["gate_epochs"]):
        generator = torch.Generator().manual_seed(config["seed"] + epoch)
        order = torch.randperm(len(y), generator=generator)
        for idx in order.split(config["gate_batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = runner.confidence_head(h[idx].cuda()).squeeze(-1)
                loss = gate_loss(logits, y[idx].cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
        metrics, _ = gate_metrics(runner, dev, config)
        history.append({"epoch": epoch + 1, **metrics})
        if metrics["bce"] < best:
            best = metrics["bce"]
            checkpoint(output / "confidence.pt", runner, config, profile, data, "confidence", previous["scales"],
                       {"dev": metrics, "epoch": epoch + 1, "module_selection": previous["extra"]})
    write_json(output / "confidence_history.json", history)
    return {"status": "CONFIDENCE_TRAINED", "best_dev_bce": best, "train_positive_rate": y.mean().item(),
            "checkpoint": str(output / "confidence.pt")}


def calibrate_confidence(runner, teacher, data, config, profile, output, previous):
    features = gate_features(runner, teacher, data, config, "calibration")
    metrics, scores = gate_metrics(runner, features, config)
    result = calibrate(scores, features[1].tolist(), config["threshold_grid"], config["risk_limit"], config["minimum_accepted"])
    result["metrics"] = metrics
    write_json(output / "calibration.json", result)
    write_json(output / "calibration_scores.json", {"scores": scores, "correct": features[1].tolist()})
    checkpoint(output / "calibrated.pt", runner, config, profile, data, "calibrated", previous["scales"],
               {"calibration": result, "confidence_selection": previous["extra"]})
    return result


def run(root, config, profile, stage, source, output, plan, warm_start=None):
    root = Path(root)
    output = Path(output) if output else root / "results/training" / f"{datetime.now():%Y%m%d-%H%M%S}-{profile}-{stage}"
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(output / "plan.json", plan)
    try:
        if config.get("incremental"):
            from .incremental import ChunkData
            data = ChunkData(root, config, profile)
        else:
            data = Data(root, config, profile)
        write_json(output / "splits.json", data.splits)
        if config.get("incremental"):
            write_json(output / "chunk.json", data.chunk)
        # Bind the recorded revision to actual local weights, not merely a path label.
        weights = sorted((root / config["model_path"]).glob("*.safetensors"))
        if len(weights) != 1 or sha256(weights[0]) != data.manifest["weights_etag_sha256"]:
            raise ValueError("Model weight hash differs from the cached dataset manifest")
        provenance = {"sources": {str(p.relative_to(root)): sha256(p) for directory in ("ced_training", "exit_cost")
                                  for p in sorted((root / directory).glob("*.py"))},
                      "entrypoint_sha256": sha256(root / "scripts/train_ced.py"),
                      "checkpoint_sha256": sha256(source) if source else None,
                      "warm_start_sha256": sha256(warm_start) if warm_start else None,
                      "distilled_manifest_sha256": data.distilled_manifest_hash,
                      "incremental_manifest_sha256": getattr(data, "incremental_manifest_hash", None),
                      "torch": torch.__version__, "transformers": transformers_version,
                      "data_manifest_sha256": data.manifest_hash}
        write_json(output / "provenance.json", provenance)
        runner = load_runner(root, config)
        provenance["gpu"] = torch.cuda.get_device_name()
        write_json(output / "provenance.json", provenance)
        teacher = Teacher(runner)
        previous = None
        if stage != "modules":
            expected = {"confidence": "modules", "calibrate": "confidence", "evaluate": "calibrated"}[stage]
            previous = load_checkpoint(source, runner, config, profile, data, expected)
        if stage == "modules":
            warm_state = load_warm_start(warm_start, runner, config, profile, data) if warm_start else None
            result = train_modules(runner, teacher, data, config, profile, output, warm_state)
        elif stage == "confidence":
            result = train_confidence(runner, teacher, data, config, profile, output, previous)
        elif stage == "calibrate":
            result = calibrate_confidence(runner, teacher, data, config, profile, output, previous)
        else:
            from .evaluation import evaluate
            result = evaluate(runner, teacher, data, config, output, previous)
        torch.cuda.synchronize()
        result["elapsed_seconds"] = time.monotonic() - started
        if "supervised_tokens" in result:
            result["end_to_end_supervised_tokens_per_second"] = result["supervised_tokens"] / result["elapsed_seconds"]
        result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        result["quality_status"] = "DIAGNOSTIC_ONLY_NOT_A_GENERALIZATION_CLAIM"
        write_json(output / "result.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except BaseException:
        write_json(output / "failure.json", {"stage": stage, "elapsed_seconds": time.monotonic() - started,
                                             "traceback": traceback.format_exc()})
        raise
