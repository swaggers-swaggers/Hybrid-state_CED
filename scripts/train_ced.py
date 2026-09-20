#!/usr/bin/env python3
"""Review-first training CLI: explicit --run is required for every GPU stage."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ced_training.protocol import make_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen35_08b_train.json")
    parser.add_argument("--profile", choices=("smoke", "pilot"), default="pilot")
    parser.add_argument("--stage", choices=("modules", "confidence", "calibrate", "evaluate"), default="modules")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--warm-start", type=Path, help="Modules only: continue from natural-data pilot on generated completions")
    parser.add_argument("--output", type=Path, help="Fresh directory; never overwrite an existing run")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    plan = make_plan(config, args.profile, ROOT)
    plan["requested_stage"] = args.stage
    if not args.run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    if args.stage != "modules" and args.checkpoint is None:
        parser.error("This stage requires --checkpoint from its predecessor")
    if args.stage == "modules" and args.checkpoint is not None:
        parser.error("modules starts a fresh trial; resuming is deliberately unsupported")
    if args.warm_start and (args.stage != "modules" or not (config.get("distilled_data_path") or config.get("incremental"))):
        parser.error("--warm-start requires generated-data or incremental modules stage")
    if args.stage == "modules" and (config.get("distilled_data_path") or config.get("incremental")) and not args.warm_start:
        parser.error("Continuation requires --warm-start")
    from ced_training.engine import run
    run(ROOT, config, args.profile, args.stage, args.checkpoint, args.output, plan, warm_start=args.warm_start)


if __name__ == "__main__":
    main()
