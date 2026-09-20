"""One exact 1M-target continuation chunk; other splits stay fixed for comparison."""
import json
from pathlib import Path
import random
import numpy as np
import torch
from .engine import Data
from .corpus import chunk_spans
from .protocol import SEMANTICS, sha256


def masked_prediction_rows(captured, tokens, mask):
    if mask.shape != tokens.shape or bool(mask[:, -1].any()):
        raise ValueError("Mask shape/boundary differs from the token window")
    valid = mask[:, :-1].bool()
    if not bool(valid.any()):
        raise ValueError("Empty supervision batch")
    return ({name: value[:, :-1][valid] for name, value in captured.items()}, tokens[:, 1:][valid])


class ChunkData(Data):
    def __init__(self, root, config, profile):
        super().__init__(root, config, profile)
        if self.distilled:
            raise ValueError("Fresh natural-text chunks cannot also use generated-data mode")
        spec = config["incremental"]
        path = root / spec["data_path"]
        manifest_path = path / "manifest.json"
        m = json.loads(manifest_path.read_text())
        self.incremental_manifest_hash = sha256(manifest_path)
        if (m["status"] != "COMPLETE_DATA_ONLY_NO_TRAINING" or m["sequence_length"] != 256
                or m["model_revision"] != self.manifest["model_revision"]
                or m["freshness"]["prior_manifest_sha256"] != self.manifest_hash
                or m["effective_next_token_targets"] != 10_000_000):
            raise ValueError("Fresh corpus provenance differs from the audited 10M dataset")
        for name, info in m["files"].items():
            p = path / name
            if p.stat().st_size != info["bytes"] or sha256(p) != info["sha256"]:
                raise ValueError(f"Corrupt fresh corpus file: {name}")
        for name, digest in m["tokenizer_files"].items():
            if sha256(root / config["model_path"] / name) != digest:
                raise ValueError("Fresh corpus tokenizer differs")
        n = m["num_sequences"]
        self.fresh_tokens = np.memmap(path / "train.int32.bin", dtype="<i4", mode="r").reshape(n, 256)
        mask = np.memmap(path / "loss_mask.uint8.bin", dtype="uint8", mode="r").reshape(n, 256)
        if (mask.max() > 1 or mask[:, -1].any() or int(mask.sum()) != m["effective_next_token_targets"]
                or self.fresh_tokens.min() < 0 or self.fresh_tokens.max() >= 248320):
            raise ValueError("Fresh token/mask contents invalid")
        start = spec["chunk_index"] * spec["chunk_tokens"]
        spans = chunk_spans(mask.sum(axis=1).tolist(), start, spec["chunk_tokens"])
        self.chunk_masks = {}
        for row, first, end in spans:
            active = np.flatnonzero(mask[row])[first:end]
            local = np.zeros(256, dtype="uint8")
            local[active] = 1
            self.chunk_masks[row] = local
        if sum(int(x.sum()) for x in self.chunk_masks.values()) != spec["chunk_tokens"]:
            raise AssertionError("Chunk mask count differs from the exact budget")
        self.splits["modules"] = {"split":"fresh_train", "indices":[r for r, _, _ in spans]}
        self.chunk = {"chunk_index":spec["chunk_index"], "start_effective_target":start,
                      "end_effective_target_exclusive":start + spec["chunk_tokens"],
                      "effective_targets":spec["chunk_tokens"], "windows":len(spans),
                      "spans":[list(x) for x in spans], "manifest_sha256":self.incremental_manifest_hash}

    def batches(self, name, batch_size, *, epoch=None, limit=None):
        if name != "modules":
            yield from super().batches(name, batch_size, epoch=epoch, limit=limit)
            return
        indices = list(self.splits["modules"]["indices"])
        if limit is not None:
            indices = indices[:limit]
        if epoch is not None:
            random.Random(87231 + epoch).shuffle(indices)
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            tokens = torch.from_numpy(np.array(self.fresh_tokens[selected], dtype=np.int64)).cuda()
            mask = torch.from_numpy(np.stack([self.chunk_masks[i] for i in selected])).cuda()
            yield tokens, mask

    def rows(self, captured, batch, split):
        if split == "modules":
            tokens, mask = batch
            return masked_prediction_rows(captured, tokens, mask)
        return super().rows(captured, batch, split)


def load_chunk_warm_start(path, runner, config, profile, data):
    spec = config["incremental"]
    if sha256(path) != spec["source_checkpoint_sha256"]:
        raise ValueError("Warm-start checkpoint hash differs from the reviewed run config")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (state["schema_version"] != 1 or state["semantics"] != SEMANTICS or state["stage"] != "modules"
            or state["profile"] != profile or state["data_manifest_sha256"] != data.manifest_hash
            or state["model_revision"] != data.manifest["model_revision"]
            or state["model_weights_sha256"] != data.manifest["weights_etag_sha256"]):
        raise ValueError("Incompatible main-module checkpoint")
    ignore = {"distilled_data_path", "incremental", "eval_every_steps"}
    if ({k:v for k,v in state["config"].items() if k not in ignore}
            != {k:v for k,v in config.items() if k not in ignore}):
        raise ValueError("Architecture/loss/optimizer settings differ from the previous trial")
    for split in ("gate", "dev", "calibration", "test"):
        if state["split_indices"][split] != data.splits[split]:
            raise ValueError("Held-out split changed during continuation")
    source_chunk = state.get("chunk")
    if spec["chunk_index"] == 0:
        if source_chunk is not None:
            raise ValueError("First fresh chunk requires a checkpoint preceding the fresh corpus")
    elif (source_chunk is None or source_chunk["chunk_index"] != spec["chunk_index"] - 1
          or source_chunk["end_effective_target_exclusive"] != data.chunk["start_effective_target"]
          or source_chunk["manifest_sha256"] != data.incremental_manifest_hash):
        raise ValueError("Chunks must continue consecutively from the same corpus without repeated targets")
    for name in ("readout_map", "kv_projectors"):
        getattr(runner, name).load_state_dict(state["modules"][name], strict=True)
    return state
