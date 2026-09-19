#!/usr/bin/env python3
"""Print a CPU-only plan by default. --run explicitly loads the model and benchmarks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exit_cost.protocol import make_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen35_08b_exit_cost.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Print workload and estimate; never imports torch.")
    mode.add_argument("--run", action="store_true", help="After human review: load local model, validate, and measure.")
    parser.add_argument("--output", type=Path, help="New output directory; existing directories are never overwritten.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    plan = make_plan(config)
    if not args.run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    from exit_cost.runtime import run
    run(config, ROOT, args.output)


if __name__ == "__main__":
    main()
