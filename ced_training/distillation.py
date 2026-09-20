"""Generate completion-only training examples from the unchanged original model."""
from __future__ import annotations
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from transformers import DynamicCache
from .engine import Data, load_runner, write_json
from .protocol import sha256


@torch.inference_mode()
def generate(root, config, profile, output, batch_size=8):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        if config.get("distilled_data_path"):
            raise ValueError("Generation must use the unmodified natural-data configuration")
        data = Data(root, config, profile)
        weights = sorted((root / config["model_path"]).glob("*.safetensors"))
        if len(weights) != 1 or sha256(weights[0]) != data.manifest["weights_etag_sha256"]:
            raise ValueError("Original teacher weights differ from the source manifest")
        runner = load_runner(root, config)
        eos = runner.model.generation_config.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
        pad = runner.model.generation_config.pad_token_id
        pad = pad if pad is not None else (eos_ids[0] if eos_ids else 0)
        indices = data.splits["modules"]["indices"]
        partial = output / "completions.partial.bin"
        stored = np.memmap(partial, dtype="<i4", mode="w+", shape=(len(indices), 512))
        total_generated, eos_stopped = 0, 0
        lengths_all = []
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            prompt = torch.from_numpy(np.array(data.arrays["train"][selected], dtype=np.int64)).cuda()
            batch = len(selected)
            cache = DynamicCache(config=runner.model.config)
            token = runner.native(prompt, cache).argmax(-1)
            completion = torch.full((batch, 256), pad, device=prompt.device, dtype=torch.long)
            done = torch.zeros(batch, device=prompt.device, dtype=torch.bool)
            lengths = torch.zeros(batch, device=prompt.device, dtype=torch.long)
            for position in range(256):
                token = torch.where(done[:, None], pad, token)
                completion[:, position] = token[:, 0]
                lengths += (~done).long()
                if eos_ids:
                    is_eos = torch.zeros_like(done)
                    for eos_id in eos_ids:
                        is_eos |= token[:, 0] == eos_id
                    done |= is_eos
                if bool(done.all()) or position == 255:
                    break
                token = runner.native(token, cache).argmax(-1)
            stored[start:start + batch, :256] = prompt.cpu().numpy()
            stored[start:start + batch, 256:] = completion.cpu().numpy()
            local_lengths = lengths.cpu().tolist()
            lengths_all.extend(local_lengths)
            total_generated += sum(local_lengths)
            eos_stopped += int(done.sum().item())
            with (output / "generation.jsonl").open("a") as log:
                log.write(json.dumps({"start": start, "count": batch, "completion_lengths": local_lengths}) + "\n")
            del cache, prompt, token, completion
        stored.flush()
        del stored
        complete = output / "completions.int32.bin"
        partial.replace(complete)
        torch.cuda.synchronize()
        manifest = {"schema_version": 1, "status": "COMPLETE", "strategy": "greedy_stop_at_eos",
                    "teacher": config["model_path"], "model_weights_sha256": data.manifest["weights_etag_sha256"],
                    "source_manifest_sha256": data.manifest_hash, "profile": profile,
                    "prompt_source": "train only, modules partition", "prompt_indices": indices,
                    "prompt_length": 256, "max_new_tokens": 256, "sequence_length": 512,
                    "eos_token_ids": eos_ids, "pad_token_id": pad, "completion_lengths": lengths_all,
                    "generated_tokens": total_generated, "eos_stopped_sequences": eos_stopped,
                    "file": complete.name, "sha256": sha256(complete), "batch_size": batch_size,
                    "elapsed_seconds": time.monotonic() - started,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                    "generator_sha256": sha256(Path(__file__)),
                    "sources": {str(p.relative_to(root)): sha256(p) for d in ("ced_training", "exit_cost") for p in sorted((root / d).glob("*.py"))}}
        write_json(output / "manifest.json", manifest)
        print(json.dumps({k: manifest[k] for k in ("status", "generated_tokens", "eos_stopped_sequences", "elapsed_seconds", "peak_allocated_mib")}, indent=2))
    except BaseException:
        write_json(output / "failure.json", {"traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic() - started})
        raise
