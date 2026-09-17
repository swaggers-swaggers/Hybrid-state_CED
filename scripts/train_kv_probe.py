#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from transformers import Qwen3_5ForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1.io import TokenSequenceStore, checkpoint_metadata, load_settings  # noqa: E402
from phase1.metrics import normalized_mse  # noqa: E402
from phase1.models import AsymmetricFusionKVProbe, build_trainable_probes, parameter_count  # noqa: E402
from phase1.teacher import QwenFeatureCapture, teacher_raw_kv, TARGET_SEMANTICS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 1 layer-19 K/V probes")
    parser.add_argument("--config", default="configs/qwen35_08b_phase1.yaml")
    parser.add_argument("--max-train-sequences", type=int)
    parser.add_argument("--target-layer", type=int)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: Path,
    probes: torch.nn.ModuleDict,
    random_probe: torch.nn.Module,
    settings: dict[str, Any],
    step: int,
    train_tokens: int,
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fusion = probes["multi_layer_fusion"]
    assert isinstance(fusion, AsymmetricFusionKVProbe)
    torch.save(
        {
            "schema_version": 2,
            "target_semantics": TARGET_SEMANTICS,
            "step": step,
            "train_tokens": train_tokens,
            "elapsed_seconds": elapsed_seconds,
            "settings": settings,
            "model": checkpoint_metadata(settings["model_path"]),
            "probe_state_dict": probes.state_dict(),
            "random_probe_state_dict": random_probe.state_dict(),
            "parameter_counts": {name: parameter_count(probe) for name, probe in probes.items()},
            "fusion_weights": fusion.fusion_weights(),
        },
        path,
    )


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
        raise RuntimeError("Phase 1 training requires CUDA")
    set_seed(settings["seed"])
    device = torch.device(settings["device"])
    dtype = getattr(torch, settings["dtype"])
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        settings["model_path"], dtype=dtype, local_files_only=True
    ).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    text_config = model.config.text_config
    target_layer = settings["probe"]["target_layer"]
    source_layers = settings["probe"]["source_layers"]
    attention = model.model.language_model.layers[target_layer].self_attn
    kv_size = text_config.num_key_value_heads * text_config.head_dim

    # Probe initialization is seeded after loading the teacher so it is reproducible
    # across Transformers loading implementations.
    set_seed(settings["seed"])
    probes = build_trainable_probes(
        text_config.hidden_size,
        kv_size,
        source_layers,
        settings["probe"]["low_ranks"],
    ).to(device)
    random_probe = type(probes["trained_linear"])(text_config.hidden_size, kv_size).to(device).eval()
    for parameter in random_probe.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        probes.parameters(),
        lr=settings["probe"]["learning_rate"],
        weight_decay=settings["probe"]["weight_decay"],
    )
    store = TokenSequenceStore(settings["data_dir"], "train")
    max_sequences = args.max_train_sequences or store.num_sequences
    max_sequences = min(max_sequences, store.num_sequences)
    batch_size = settings["probe"]["batch_size"]
    total_steps = math.ceil(max_sequences / batch_size)
    warmup_steps = min(settings["probe"]["warmup_steps"], max(total_steps // 10, 1))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.jsonl"
    checkpoint_path = Path(settings["checkpoint_path"])
    start_time = time.perf_counter()
    processed_sequences = 0

    with log_path.open("w", encoding="utf-8") as log_file, QwenFeatureCapture(
        model, source_layers, target_layer
    ) as capture:
        for step, input_ids in enumerate(
            store.batches(batch_size, device, shuffle=True, seed=settings["seed"], max_sequences=max_sequences),
            start=1,
        ):
            sources, target_hidden = capture.capture(input_ids)
            settings["target_cache_verification"] = capture.cache_verification
            with torch.inference_mode():
                target_k, target_v = teacher_raw_kv(attention, target_hidden)

            optimizer.zero_grad(set_to_none=True)
            losses: dict[str, torch.Tensor] = {}
            with torch.autocast(device_type="cuda", dtype=dtype):
                for name, probe in probes.items():
                    predicted_k, predicted_v = probe(sources)
                    losses[name] = normalized_mse(predicted_k, target_k) + normalized_mse(predicted_v, target_v)
                total_loss = torch.stack(list(losses.values())).sum()
            total_loss.backward()
            grad_norm = clip_grad_norm_(probes.parameters(), settings["probe"]["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            processed_sequences += input_ids.shape[0]

            if step == 1 or step % settings["probe"]["log_every"] == 0 or step == total_steps:
                record = {
                    "step": step,
                    "total_steps": total_steps,
                    "tokens": processed_sequences * store.sequence_length,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                    "losses": {name: float(loss.detach()) for name, loss in losses.items()},
                    "elapsed_seconds": time.perf_counter() - start_time,
                }
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_file.flush()
                print(json.dumps(record, ensure_ascii=False), flush=True)
            if step % 1000 == 0:
                save_checkpoint(
                    checkpoint_path,
                    probes,
                    random_probe,
                    settings,
                    step,
                    processed_sequences * store.sequence_length,
                    time.perf_counter() - start_time,
                )

    elapsed = time.perf_counter() - start_time
    save_checkpoint(
        checkpoint_path,
        probes,
        random_probe,
        settings,
        total_steps,
        processed_sequences * store.sequence_length,
        elapsed,
    )
    print(
        json.dumps(
            {"checkpoint": str(checkpoint_path), "steps": total_steps, "tokens": processed_sequences * store.sequence_length, "elapsed_seconds": elapsed},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
