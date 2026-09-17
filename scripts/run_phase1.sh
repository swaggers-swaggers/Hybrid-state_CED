#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHONPATH="$PROJECT_DIR/.deps${PYTHONPATH:+:$PYTHONPATH}" \
HF_HOME="$PROJECT_DIR/.cache/huggingface" \
conda run -n ai python scripts/prepare_phase1_data.py --config configs/qwen35_08b_phase1.yaml

# The runner records scientific NO-GO outcomes without skipping remaining layers;
# actual execution/test failures still stop the run.
exec conda run -n ai python scripts/run_phase1_corrected.py
