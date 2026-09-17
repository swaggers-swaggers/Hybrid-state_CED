#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
from pathlib import Path

import torch
import transformers


def main() -> int:
    model_path = Path(__file__).resolve().parents[1] / "models" / "Qwen3.5-0.8B-Base"
    payload = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "model_path": str(model_path),
        "model_present": (model_path / "model.safetensors.index.json").is_file(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        payload["gpu"] = properties.name
        payload["gpu_memory_bytes"] = properties.total_memory
        payload["compute_capability"] = f"{properties.major}.{properties.minor}"
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["model_present"] and payload["cuda_available"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
