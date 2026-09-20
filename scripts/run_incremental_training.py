#!/usr/bin/env python3
"""Run exactly one 1M-target chunk and its final evaluations; never auto-advance."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ced_training.protocol import make_plan, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-index", type=int, default=0, help="Zero-based 1M-target interval in the 10M corpus")
    parser.add_argument("--warm-start", type=Path, default=ROOT / "results/training/20260919-175254-pilot-distillation/distilled/modules/modules.pt")
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/qwen35_08b_train.json").read_text())
    # Main-module development evaluation only before/after this chunk, not every 512 steps.
    config["eval_every_steps"] = 1_000_000
    config["incremental"] = {"data_path":"data/wikitext103_10000000_fresh_20260919-222037",
                             "chunk_index":args.chunk_index,"chunk_tokens":1_000_000,
                             "source_checkpoint_sha256":sha256(args.warm_start)}
    plan = make_plan(config, "pilot", ROOT)
    plan["expected_minutes_not_measured"] = [12,20]
    if not args.run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    output = args.output or ROOT / "results/training" / f"{datetime.now():%Y%m%d-%H%M%S}-fresh-chunk{args.chunk_index + 1:02d}-1m"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config, indent=2))
    (output / "plan.json").write_text(json.dumps(plan, indent=2))
    source_hashes = {str(p.relative_to(ROOT)):sha256(p) for directory in ("ced_training", "exit_cost", "scripts") for p in (ROOT / directory).glob("*.py")}
    (output / "source_hashes.json").write_text(json.dumps(source_hashes, indent=2))
    record = {"status":"RUNNING", "output":str(output), "chunk_index":args.chunk_index,
              "effective_targets":1_000_000, "warm_start":str(args.warm_start), "stages":[]}
    (output / "execution.json").write_text(json.dumps(record, indent=2))
    begin = time.monotonic()
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    with (output / "gpu_before.txt").open("w") as log:
        subprocess.run(["nvidia-smi"], stdout=log, stderr=subprocess.STDOUT, check=False)
    previous = None
    try:
        for stage, filename in (("modules","modules.pt"),("confidence","confidence.pt"),("calibrate","calibrated.pt"),("evaluate",None)):
            if any(sha256(ROOT / name) != digest for name,digest in source_hashes.items()):
                raise RuntimeError("Training source changed during the run")
            command = [sys.executable,str(ROOT / "scripts/train_ced.py"),"--run","--profile","pilot","--stage",stage,
                       "--config",str(config_path),"--output",str(output / stage)]
            if stage == "modules":
                command += ["--warm-start",str(args.warm_start)]
            else:
                command += ["--checkpoint",str(previous)]
            start = time.monotonic()
            with (output / f"{stage}.log").open("w") as log:
                result = subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            record["stages"].append({"stage":stage,"returncode":result.returncode,"wall_seconds":time.monotonic()-start})
            if result.returncode:
                raise RuntimeError(f"{stage} failed; preserve results and stop dependent stages")
            if filename:
                previous = output / stage / filename
        record["status"] = "COMPLETED"
    except BaseException as exc:
        record["status"], record["error"] = "FAILED", str(exc)
        raise
    finally:
        record["wall_seconds"] = time.monotonic() - begin
        (output / "execution.json").write_text(json.dumps(record,indent=2))
        print(json.dumps(record,indent=2),flush=True)


if __name__ == "__main__":
    main()
