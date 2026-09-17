from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from phase1.metrics import bootstrap_relative_improvement, gate_decision, normalized_mse
from phase1.models import AsymmetricFusionKVProbe, LowRankKVProbe, build_trainable_probes, parameter_count
from phase1.teacher import require_corrected_checkpoint, TARGET_SEMANTICS


ROOT = Path(__file__).resolve().parents[1]


class ProbeShapeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sources = {
            3: torch.randn(2, 7, 16),
            7: torch.randn(2, 7, 16),
            11: torch.randn(2, 7, 16),
        }

    def test_probe_output_shapes(self) -> None:
        probes = build_trainable_probes(16, 8, [3, 7, 11], [2, 4])
        for probe in probes.values():
            key, value = probe(self.sources)
            self.assertEqual(key.shape, (2, 7, 8))
            self.assertEqual(value.shape, (2, 7, 8))

    def test_low_rank_parameter_count(self) -> None:
        probe = LowRankKVProbe(16, 8, 4)
        self.assertEqual(parameter_count(probe), 2 * (16 * 4 + 4 * 8))

    def test_asymmetric_weights_normalize(self) -> None:
        probe = AsymmetricFusionKVProbe(16, 8, [3, 7, 11])
        weights = probe.fusion_weights()
        self.assertAlmostEqual(sum(weights["k"]), 1.0, places=6)
        self.assertAlmostEqual(sum(weights["v"]), 1.0, places=6)


class MetricTest(unittest.TestCase):
    def test_legacy_checkpoint_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, 'retrain'):
            require_corrected_checkpoint({'schema_version': 1})
        require_corrected_checkpoint({'target_semantics': TARGET_SEMANTICS})

    def test_nmse(self) -> None:
        target = torch.tensor([1.0, 2.0])
        self.assertEqual(normalized_mse(target, target).item(), 0.0)
        self.assertAlmostEqual(normalized_mse(torch.zeros_like(target), target).item(), 1.0)

    def test_paired_bootstrap(self) -> None:
        result = bootstrap_relative_improvement([1.0] * 20, [0.5] * 20, 100, 7)
        self.assertAlmostEqual(result["mean"], 0.5)
        self.assertAlmostEqual(result["ci95_low"], 0.5)

    def test_gate_uses_validation_selected_method(self) -> None:
        methods = ["original_projection", "random_linear", "trained_linear", "low_rank_2"]
        metrics = {}
        batches = {}
        for name in methods:
            value = {"original_projection": 1.0, "random_linear": 2.0, "trained_linear": 0.5, "low_rank_2": 0.7}[name]
            metrics[name] = {"k_nmse": value, "v_nmse": value, "attention_output_nmse": value}
            batches[name] = {"attention_output_nmse": [value] * 20}
        decision = gate_decision(metrics, batches, 100, 0.1, 0.0, 7, selected_method="trained_linear")
        self.assertEqual(decision["status"], "GO")
        self.assertEqual(decision["best_method"], "trained_linear")


class Phase1ArtifactTest(unittest.TestCase):
    def test_corrected_artifacts_have_real_cache_evidence(self) -> None:
        base = ROOT / 'results/phase1_corrected'
        if not (base / 'phase1_all_layers.json').exists():
            self.skipTest('Corrected full run not produced yet')
        for layer in (15,19,23):
            result = json.loads((base / f'layer{layer}/phase1_results.json').read_text())
            self.assertEqual(result['target_semantics'], TARGET_SEMANTICS)
            self.assertEqual(result['checkpoint']['train_tokens'], 4_999_936)
            for split in ('validation','test'):
                self.assertTrue(result['splits'][split]['target_cache_verification']['exact_equal'])
            selected = min((n for n in result['method_order'] if n not in ('zero','original_projection','random_linear')), key=lambda n: result['splits']['validation']['metrics'][n]['attention_output_nmse'])
            self.assertEqual(result['selected_method'], selected)

    def test_data_manifest_has_independent_splits(self) -> None:
        manifest = json.loads((ROOT / "data" / "phase1_wikitext103" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(set(manifest["splits"]), {"train", "validation", "test"})
        hashes = {split["sha256"] for split in manifest["splits"].values()}
        self.assertEqual(len(hashes), 3)


if __name__ == "__main__":
    unittest.main()
