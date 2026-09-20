#!/usr/bin/env python3
"""Prepare fresh natural-text data only. Never imports a model or starts training."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ced_training.corpus import article_title, title_key, complete_articles, loss_mask
from ced_training.protocol import sha256
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare(source, output, target_count):
    # Tokenizer and array operations only, even on hosts with CUDA available.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import numpy as np
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    try:
        source = source.resolve()
        if source.name != REVISION:
            raise ValueError("Require the pinned official dataset snapshot directory")
        shards = sorted((source / "wikitext-103-v1").glob("train-*.parquet"))
        if not shards:
            raise ValueError("No downloaded wikitext-103-v1 train Parquet files")
        prior_path = ROOT / "data/phase1_wikitext103"
        prior = json.loads((prior_path / "manifest.json").read_text())
        tokenizer_path = ROOT / "models/Qwen3.5-0.8B-Base"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        eos = tokenizer.eos_token_id
        if eos is None:
            raise ValueError("Source tokenization requires EOS between nonempty records")
        old_arrays, excluded_windows = {}, set()
        for name, info in prior["splits"].items():
            path = prior_path / info["file"]
            if sha256(path) != info["sha256"] or path.stat().st_size != info["num_tokens"] * 4:
                raise ValueError(f"Cached {name} source no longer matches its manifest")
            old_arrays[name] = np.memmap(path, dtype="<i4", mode="r").reshape(-1, 256)
            for row in old_arrays[name]:
                excluded_windows.add(hashlib.sha256(row.tobytes()).digest())

        def rows():
            index = 0
            for shard in shards:
                for batch in pq.ParquetFile(shard).iter_batches(batch_size=4096, columns=["text"]):
                    for text in batch.column(0).to_pylist():
                        yield index, text
                        index += 1

        stream = iter(rows())
        old_flat = old_arrays["train"].reshape(-1)
        old_titles, buffer = set(), []
        verified_tokens, nonempty, last_old_row = 0, 0, None
        limit = prior["splits"]["train"]["nonempty_records_consumed"]

        def verify_buffer(texts):
            nonlocal verified_tokens
            encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]
            flat = np.asarray([token for ids in encoded for token in (*ids, eos)], dtype="<i4")
            take = min(len(flat), len(old_flat) - verified_tokens)
            if take and not np.array_equal(flat[:take], old_flat[verified_tokens:verified_tokens + take]):
                raise ValueError("Pinned raw source/tokenizer does not reproduce existing token cache; cannot certify freshness")
            verified_tokens += take

        for row_id, text in stream:
            if not text.strip():
                continue
            title = article_title(text)
            if title is not None:
                old_titles.add(title_key(title))
            buffer.append(text)
            nonempty += 1
            last_old_row = row_id
            if len(buffer) == 256:
                verify_buffer(buffer)
                buffer.clear()
            if nonempty == limit:
                break
        if buffer:
            verify_buffer(buffer)
        if nonempty != limit or verified_tokens != len(old_flat):
            raise ValueError("Could not verify the entire previously consumed training prefix")

        accepted, targets, considered_articles = 0, 0, 0
        seen_windows, seen_titles = set(), set(old_titles)
        duplicate_windows, repeated_titles = 0, 0
        article_count, discarded_short_tail_tokens = 0, 0
        token_path, mask_path, id_path = (output / n for n in ("train.int32.bin", "loss_mask.uint8.bin", "window_article_ids.int32.bin"))
        with token_path.open("wb") as token_file, mask_path.open("wb") as mask_file, id_path.open("wb") as id_file, (output / "articles.jsonl").open("w") as article_file:
            for article in complete_articles(stream):
                considered_articles += 1
                key = title_key(article["title"])
                if key in seen_titles:
                    repeated_titles += 1
                    continue
                seen_titles.add(key)
                encoded = []
                for start in range(0, len(article["texts"]), 256):
                    batch_ids = tokenizer(article["texts"][start:start + 256], add_special_tokens=False, return_attention_mask=False)["input_ids"]
                    for ids in batch_ids:
                        encoded.extend(ids)
                        encoded.append(eos)
                token_array = np.asarray(encoded, dtype="<i4")
                if len(token_array) and (token_array.min() < 0 or token_array.max() >= 248320):
                    raise ValueError("Out-of-range token ID")
                start_window, start_targets = accepted, targets
                for offset in range(0, len(token_array) - 255, 256):
                    window = token_array[offset:offset + 256]
                    digest = hashlib.sha256(window.tobytes()).digest()
                    if digest in excluded_windows or digest in seen_windows:
                        duplicate_windows += 1
                        continue
                    seen_windows.add(digest)
                    mask = np.asarray(loss_mask(target_count - targets), dtype="uint8")
                    token_file.write(window.tobytes())
                    mask_file.write(mask.tobytes())
                    id_file.write(np.asarray([article_count], dtype="<i4").tobytes())
                    targets += int(mask.sum())
                    accepted += 1
                    if targets == target_count:
                        break
                discarded_short_tail_tokens += len(token_array) % 256
                if accepted > start_window:
                    record = {"article_id": article_count, "title": article["title"], "source_first_row": article["first_row"],
                              "source_last_row": article["last_row"], "nonempty_records": len(article["texts"]),
                              "text_sha256": hashlib.sha256(json.dumps(article["texts"], ensure_ascii=False).encode()).hexdigest(),
                              "window_start": start_window, "window_end_exclusive": accepted, "effective_targets": targets - start_targets}
                    article_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    article_count += 1
                if targets == target_count:
                    break
        if targets != target_count:
            raise ValueError(f"Source exhausted with {targets} targets, requested {target_count}")
        # Independent readback catches packing/mask/size errors, without loading a model.
        final_tokens = np.memmap(token_path, dtype="<i4", mode="r").reshape(-1, 256)
        final_mask = np.memmap(mask_path, dtype="uint8", mode="r").reshape(-1, 256)
        final_ids = np.memmap(id_path, dtype="<i4", mode="r")
        if (len(final_tokens) != math.ceil(target_count / 255) or len(final_ids) != len(final_tokens)
                or int(final_mask.sum()) != target_count or final_mask.max() > 1 or final_mask[:, -1].any()
                or final_tokens.min() < 0 or final_tokens.max() >= 248320
                or int(final_ids.min()) != 0 or int(final_ids.max()) != article_count - 1):
            raise AssertionError("Prepared dataset readback failed")
        files = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in (token_path, mask_path, id_path, output / "articles.jsonl")}
        manifest = {"schema_version": 1, "status": "COMPLETE_DATA_ONLY_NO_TRAINING", "dataset": "Salesforce/wikitext",
                    "dataset_config": "wikitext-103-v1", "dataset_revision": REVISION, "upstream_split": "train",
                    "source_url": "https://huggingface.co/datasets/Salesforce/wikitext", "source_files": [
                        {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p), "rows": pq.ParquetFile(p).metadata.num_rows} for p in shards],
                    "tokenizer": str(tokenizer_path), "model_revision": prior["model_revision"],
                    "tokenizer_files": {p.name: sha256(p) for p in (tokenizer_path / "tokenizer.json", tokenizer_path / "tokenizer_config.json")},
                    "sequence_length": 256, "num_sequences": accepted, "input_tokens": accepted * 256,
                    "effective_next_token_targets": targets, "articles": article_count, "eos_token_id": eos,
                    "encoding": "Nonempty source records unchanged, each tokenized without added special tokens, then EOS appended. Windows never cross article boundaries.",
                    "loss_mask": "uint8 [N,256], position t=1 supervises token[t+1]; last position always 0; final row is truncated by mask to meet exact budget.",
                    "packing": "Non-overlapping complete 256-token windows within each article; short article tails discarded; no synthetic generation or repeated epochs.",
                    "freshness": {"prior_manifest_sha256": sha256(prior_path / "manifest.json"), "verified_prefix_tokens": verified_tokens,
                                  "skipped_nonempty_records": nonempty, "last_skipped_source_row": last_old_row,
                                  "policy": "Skip entire cached prefix and its boundary article; exclude repeated article titles and exact windows matching any existing split or this output.",
                                  "excluded_prior_article_titles": len(old_titles), "skipped_repeated_article_titles": repeated_titles,
                                  "skipped_duplicate_windows": duplicate_windows, "considered_articles": considered_articles},
                    "discarded_short_article_tail_tokens": discarded_short_tail_tokens, "files": files,
                    "elapsed_seconds": time.monotonic() - started, "script_sha256": sha256(Path(__file__)),
                    "limitations": "Natural English text only. Existing dev/calibration/test caches unchanged. This dataset is not yet wired into the current training loader; no training was started."}
        write_json(output / "manifest.json", manifest)
        print(json.dumps({"output": str(output), **{k:manifest[k] for k in ("status","num_sequences","input_tokens","effective_next_token_targets","articles","elapsed_seconds")}}, ensure_ascii=False, indent=2))
    except BaseException:
        write_json(output / "failure.json", {"traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / ".cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots" / REVISION)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokens", type=int, default=10_000_000, help="Exact number of effective next-token supervision positions")
    parser.add_argument("--prepare", action="store_true", help="Prepare data on CPU; never starts model training")
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("tokens must be positive")
    if not args.prepare:
        print(json.dumps({"status":"PLAN_ONLY_DATA_PREPARATION", "effective_tokens":args.tokens,
                          "windows":math.ceil(args.tokens/255), "dataset_revision":REVISION,
                          "source":str(args.source), "training":False},indent=2))
        return
    output = args.output or ROOT / "data" / f"wikitext103_{args.tokens}_fresh_{datetime.now():%Y%m%d-%H%M%S}"
    prepare(args.source, output, args.tokens)


if __name__ == "__main__":
    main()
