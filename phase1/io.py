from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_settings(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("model_path", "data_dir", "output_dir", "checkpoint_path"):
        settings[key] = str((PROJECT_ROOT / settings[key]).resolve())
    settings["config_path"] = str(path.resolve())
    return settings


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class TokenSequenceStore:
    def __init__(self, data_dir: str | Path, split: str) -> None:
        directory = Path(data_dir)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        split_info = manifest["splits"][split]
        self.path = directory / split_info["file"]
        self.sequence_length = int(manifest["sequence_length"])
        self.num_sequences = int(split_info["num_sequences"])
        self.tokens = np.memmap(
            self.path,
            dtype=np.int32,
            mode="r",
            shape=(self.num_sequences, self.sequence_length),
        )

    def batches(
        self,
        batch_size: int,
        device: torch.device,
        *,
        shuffle: bool,
        seed: int,
        max_sequences: int | None = None,
    ) -> Iterator[torch.Tensor]:
        count = self.num_sequences if max_sequences is None else min(self.num_sequences, max_sequences)
        indices = np.arange(self.num_sequences)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        indices = indices[:count]
        for start in range(0, count, batch_size):
            selected = indices[start : start + batch_size]
            batch = np.asarray(self.tokens[selected], dtype=np.int64)
            yield torch.from_numpy(batch).to(device=device, non_blocking=True)


def checkpoint_metadata(model_path: str | Path) -> dict[str, str]:
    metadata_path = (
        Path(model_path)
        / ".cache"
        / "huggingface"
        / "download"
        / "model.safetensors-00001-of-00001.safetensors.metadata"
    )
    try:
        revision, etag, *_ = metadata_path.read_text(encoding="utf-8").splitlines()
        return {"model_revision": revision, "weights_etag_sha256": etag}
    except (OSError, ValueError):
        return {"model_revision": "unavailable", "weights_etag_sha256": "unavailable"}
