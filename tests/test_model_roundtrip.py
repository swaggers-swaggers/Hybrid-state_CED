from __future__ import annotations

import os
import unittest
from pathlib import Path

import torch
from transformers import Qwen3_5ForConditionalGeneration

from scripts.trace_state_map import verify_boundary, verify_cache_roundtrip, verify_causality
from phase1.teacher import QwenFeatureCapture


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "Qwen3.5-0.8B-Base"


@unittest.skipUnless(os.environ.get("CED_RUN_MODEL_TESTS") == "1", "set CED_RUN_MODEL_TESTS=1 for GPU model tests")
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class RealModelRoundTripTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16, local_files_only=True
        ).eval().cuda()

    def test_teacher_cache_roundtrip(self) -> None:
        result = verify_cache_roundtrip(self.model, 16, 1e-4)
        self.assertEqual(result["status"], "PASS", result)

    def test_boundary_token(self) -> None:
        result = verify_boundary(self.model, 16, 0.25, 0.001, 0.999)
        self.assertEqual(result["status"], "PASS", result)

    def test_causal_path(self) -> None:
        result = verify_causality(self.model, 16, 1e-4)
        self.assertEqual(result["status"], "PASS", result)

    def test_phase1_targets_match_real_normalized_cache(self) -> None:
        lm = self.model.model.language_model
        for target in (15, 19, 23):
            raw = {}
            def hook(module, args):
                raw['hidden'] = args[0].detach()
            handle = lm.layers[target].register_forward_pre_hook(hook)
            try:
                with QwenFeatureCapture(self.model, [3,7,11], target) as capture:
                    for length in (7, 32):
                        tokens = torch.arange(1, length+1, device='cuda')[None]
                        _, hidden = capture.capture(tokens)
                        expected = lm.layers[target].input_layernorm(raw['hidden'])
                        self.assertTrue(torch.equal(hidden, expected))
                        self.assertFalse(torch.equal(hidden, raw['hidden']))
                    self.assertTrue(capture.cache_verification['exact_equal'])
            finally:
                handle.remove()


if __name__ == "__main__":
    unittest.main()
