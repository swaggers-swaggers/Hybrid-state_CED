"""No model/torch imports: partitions, fail-closed calibration, review-first CLI."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ced_training.protocol import calibrate, make_plan, split_indices


class TrainingProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/qwen35_08b_train.json").read_text())
        self.manifest = {"sequence_length": 256, "dataset": "Salesforce/wikitext", "dataset_config": "wikitext-103-v1",
                         "splits": {"train": {"num_sequences": 19531}, "validation": {"num_sequences": 1024}, "test": {"num_sequences": 1024}}}

    def test_fixed_split_pools_do_not_leak_or_move_between_profiles(self):
        pilot = split_indices(self.config, "pilot", self.manifest)
        smoke = split_indices(self.config, "smoke", self.manifest)
        for key in pilot:
            self.assertTrue(set(smoke[key]["indices"]) <= set(pilot[key]["indices"]))
        for left, right in (("modules", "gate"), ("dev", "calibration")):
            self.assertFalse(set(pilot[left]["indices"]) & set(pilot[right]["indices"]))
        self.assertLess(max(pilot["modules"]["indices"]), min(pilot["gate"]["indices"]))
        self.assertEqual(pilot, split_indices(self.config, "pilot", self.manifest))

    def test_distillation_preserves_held_out_windows(self):
        original = split_indices(self.config, "pilot", self.manifest)
        self.config["distilled_data_path"] = "/teacher-generated"
        self.assertEqual(original, split_indices(self.config, "pilot", self.manifest))
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / "manifest.json").write_text(json.dumps(self.manifest))
            self.config["data_path"] = str(p)
            plan = make_plan(self.config, "pilot", ROOT)
            self.assertEqual(plan["workload"]["modules"]["next_token_targets"], 4096 * 256)
            self.assertEqual(plan["workload"]["modules"]["input_tokens"], 4096 * 512)
            self.assertEqual(plan["workload"]["dev"]["next_token_targets"], 128 * 255)

    def test_pool_overflow_is_rejected(self):
        self.config["profiles"]["pilot"]["module_sequences"] = 15000
        with self.assertRaises(ValueError):
            split_indices(self.config, "pilot", self.manifest)

    def test_calibration_prefers_coverage_and_disables_on_failure(self):
        result = calibrate([.99, .95, .85, .7], [1, 1, 0, 0], [.8, .9, .99], .01, 2)
        self.assertEqual(result["threshold"], .9)
        self.assertEqual(result["selected"]["coverage"], .5)
        for scores, labels in (([.99]*5, [0]*5), ([.99], [1]), ([.1]*5, [1]*5)):
            result = calibrate(scores, labels, [.9], .01, 2)
            self.assertIsNone(result["threshold"])
        with self.assertRaises(ValueError):
            calibrate([float("nan")], [1], [.9], .01, 1)

    def test_default_cli_does_not_import_runtime_or_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "manifest.json").write_text(json.dumps(self.manifest))
            self.config["data_path"] = str(path)
            config_path = path / "config.json"
            config_path.write_text(json.dumps(self.config))
            # Sitecustomize forbids imports before invoking the real CLI.
            (path / "sitecustomize.py").write_text(
                "import sys\nclass Block:\n def find_spec(self, fullname, *args):\n"
                "  if fullname.split('.')[0] in ('torch','transformers','numpy'): raise RuntimeError('Forbidden runtime import')\n"
                "sys.meta_path.insert(0, Block())\n")
            import os
            env = dict(os.environ, PYTHONPATH=str(path), CUDA_VISIBLE_DEVICES="")
            result = subprocess.run([sys.executable, str(ROOT / "scripts/train_ced.py"), "--config", str(config_path)],
                                    capture_output=True, text=True, env=env, check=True)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["status"], "PLAN_ONLY_NO_TRAINING")
            self.assertEqual(plan["workload"]["modules"]["next_token_targets"], 4096 * 255)
            self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
