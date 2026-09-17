#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

conda run -n ai python scripts/check_environment.py
conda run -n ai python scripts/trace_state_map.py --config configs/qwen35_08b_state_map.yaml
CED_RUN_MODEL_TESTS=1 conda run -n ai python -m unittest discover -s tests -v
