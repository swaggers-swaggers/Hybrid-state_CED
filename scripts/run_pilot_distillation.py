#!/usr/bin/env python3
"""Run the approved pilot, teacher generation, then continued distillation training, sequentially."""
from __future__ import annotations
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--plan", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = ROOT / "configs/qwen35_08b_train.json"
    config = json.loads(config_path.read_text())
    plan = make_plan(config, "pilot", ROOT)
    plan.update({"pipeline": ["natural pilot: four stages", "original teacher generates up to 1048576 completion tokens", "warm-started generated-data continuation: four stages"],
                 "generation_batch_size": 8, "combined_budget_minutes_unmeasured": [60, 120],
                 "execution": "Fixed sequence; no metric-dependent tuning; stop on process failure; a failed quality gate still permits scheduled research continuation."})
    if not args.run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    output = args.output or ROOT / "results/training" / f"{datetime.now():%Y%m%d-%H%M%S}-pilot-distillation"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    started = time.monotonic()
    record = {"status": "RUNNING", "output": str(output), "stages": []}
    (output / "execution.json").write_text(json.dumps(record, indent=2))
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    source_paths = [p for d in ("ced_training", "exit_cost", "scripts") for p in (ROOT / d).glob("*.py")] + [config_path]
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    (output / "source_hashes.json").write_text(json.dumps(source_hashes, indent=2))
    with (output / "gpu_before.txt").open("w") as stream:
        subprocess.run(["nvidia-smi"], stdout=stream, stderr=subprocess.STDOUT, check=False)

    def execute(name, command):
        if any(sha256(ROOT / name) != digest for name, digest in source_hashes.items()):
            raise RuntimeError("Source changed during pipeline; stop instead of mixing implementations")
        begin = time.monotonic()
        with (output / f"{name}.log").open("w") as stream:
            result = subprocess.run([sys.executable, *command], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        record["stages"].append({"name": name, "returncode": result.returncode, "wall_seconds": time.monotonic() - begin})
        if result.returncode:
            raise RuntimeError(f"Stage failed: {name}; see its completed log")

    def training_trial(label, trial_config, warm_start=None):
        previous = None
        for stage, filename in (("modules", "modules.pt"), ("confidence", "confidence.pt"), ("calibrate", "calibrated.pt"), ("evaluate", None)):
            destination = output / label / stage
            command = [str(ROOT / "scripts/train_ced.py"), "--run", "--profile", "pilot", "--stage", stage,
                       "--config", str(trial_config), "--output", str(destination)]
            if previous:
                command += ["--checkpoint", str(previous)]
            if stage == "modules" and warm_start:
                command += ["--warm-start", str(warm_start)]
            execute(f"{label}-{stage}", command)
            if filename:
                previous = destination / filename

    try:
        training_trial("pilot", config_path)
        execute("teacher-data", [str(ROOT / "scripts/generate_distillation_data.py"), "--run", "--profile", "pilot", "--batch-size", "8", "--output", str(output / "teacher_data")])
        generated_config = dict(config, distilled_data_path=str(output / "teacher_data"))
        generated_path = output / "distilled_config.json"
        generated_path.write_text(json.dumps(generated_config, indent=2))
        training_trial("distilled", generated_path, warm_start=output / "pilot/modules/modules.pt")
        record["status"] = "COMPLETED"
    except BaseException as exc:
        record["status"] = "FAILED"
        record["error"] = str(exc)
        raise
    finally:
        record["wall_seconds"] = time.monotonic() - started
        (output / "execution.json").write_text(json.dumps(record, indent=2))
        print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
