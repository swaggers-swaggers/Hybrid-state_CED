#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1.io import checkpoint_metadata, load_settings, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fixed WikiText-103 token sequences for Phase 1")
    parser.add_argument("--config", default="configs/qwen35_08b_phase1.yaml")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_split(
    settings: dict[str, Any],
    tokenizer: Any,
    split: str,
    token_budget: int,
    output_dir: Path,
) -> dict[str, Any]:
    sequence_length = settings["dataset"]["sequence_length"]
    num_sequences = token_budget // sequence_length
    destination = output_dir / f"{split}.int32.bin"
    array = np.memmap(destination, dtype=np.int32, mode="w+", shape=(num_sequences, sequence_length))
    dataset = load_dataset(
        settings["dataset"]["name"],
        settings["dataset"]["config"],
        split=split,
        streaming=True,
    )
    buffer: list[int] = []
    written = 0
    nonempty_records = 0
    eos = tokenizer.eos_token_id
    for row in dataset:
        text = row["text"]
        if not text or not text.strip():
            continue
        nonempty_records += 1
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        buffer.extend(ids)
        if eos is not None:
            buffer.append(eos)
        while len(buffer) >= sequence_length and written < num_sequences:
            array[written] = np.asarray(buffer[:sequence_length], dtype=np.int32)
            del buffer[:sequence_length]
            written += 1
        if written == num_sequences:
            break
    array.flush()
    if written != num_sequences:
        raise RuntimeError(f"Split {split} ended after {written} sequences; expected {num_sequences}")
    return {
        "file": destination.name,
        "num_sequences": num_sequences,
        "num_tokens": num_sequences * sequence_length,
        "nonempty_records_consumed": nonempty_records,
        "sha256": sha256(destination),
    }


def main() -> int:
    settings = load_settings(parse_args().config)
    output_dir = Path(settings["data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(settings["model_path"], local_files_only=True)
    split_budgets = {
        "train": settings["dataset"]["train_tokens"],
        "validation": settings["dataset"]["validation_tokens"],
        "test": settings["dataset"]["test_tokens"],
    }
    splits = {}
    for split, budget in split_budgets.items():
        print(f"[phase1:data] preparing {split} ({budget} requested tokens)", flush=True)
        splits[split] = prepare_split(settings, tokenizer, split, budget, output_dir)
    manifest = {
        "schema_version": 1,
        "dataset": settings["dataset"]["name"],
        "dataset_config": settings["dataset"]["config"],
        "split_policy": "Upstream train/validation/test splits are preserved; sequences are packed only within each split.",
        "tokenizer": settings["model_path"],
        **checkpoint_metadata(settings["model_path"]),
        "sequence_length": settings["dataset"]["sequence_length"],
        "splits": splits,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
