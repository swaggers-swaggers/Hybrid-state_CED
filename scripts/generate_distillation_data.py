#!/usr/bin/env python3
"""Original-model generation; defaults to a model-free plan."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ced_training.protocol import make_plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/qwen35_08b_train.json")
    p.add_argument("--profile", choices=("smoke", "pilot"), default="pilot")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output", type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("batch-size must be positive")
    config = json.loads(args.config.read_text())
    plan = make_plan(config, args.profile, ROOT)
    count = plan["workload"]["modules"]["sequences"]
    if not args.run:
        print(json.dumps({"status": "PLAN_ONLY_NO_GENERATION", "sequences": count, "prompt_tokens": 256,
                          "max_generated_tokens": count * 256, "batch_size": args.batch_size,
                          "strategy": "original full model, greedy, stop at EOS, mask prompt and padding losses",
                          "source": "modules training prompts only; no development/calibration/test prompts"}, indent=2))
        return
    if args.output is None:
        p.error("--run requires a fresh --output directory")
    from ced_training.distillation import generate
    generate(ROOT, config, args.profile, args.output, args.batch_size)


if __name__ == "__main__":
    main()
