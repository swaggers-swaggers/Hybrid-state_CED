from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache

from phase0.cache_tools import assert_cache_storage_independent, clone_dynamic_cache, describe_cache
from phase0.state_map import build_static_state_map


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "Qwen3.5-0.8B-Base"


class StaticStateMapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)

    def test_official_layer_layout(self) -> None:
        state_map = build_static_state_map(self.config.text_config)
        architecture = state_map["architecture"]
        self.assertEqual(architecture["num_hidden_layers"], 24)
        self.assertEqual(architecture["linear_attention_layers"], 18)
        self.assertEqual(architecture["full_attention_layers"], 6)
        self.assertEqual(architecture["full_attention_indices"], [3, 7, 11, 15, 19, 23])

    def test_static_shapes_and_memory(self) -> None:
        state_map = build_static_state_map(self.config.text_config)
        gdn = state_map["layers"][0]["cache"]
        attention = state_map["layers"][3]["cache"]
        self.assertEqual(gdn["conv_shape"], [1, 6144, 4])
        self.assertEqual(gdn["recurrent_shape"], [1, 16, 128, 128])
        self.assertEqual(gdn["recurrent_bytes"], 524_288)
        self.assertEqual(attention["final_k_shape_formula"], [1, 2, "N", 256])
        self.assertEqual(attention["k_bytes_formula"], "1024 * N")

    def test_config_matches_downloaded_checkpoint(self) -> None:
        index = json.loads((MODEL_PATH / "model.safetensors.index.json").read_text())
        self.assertIn("model.language_model.layers.23.self_attn.k_proj.weight", index["weight_map"])


class CacheCloneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)

    def test_clone_all_heterogeneous_states(self) -> None:
        cache = DynamicCache(config=self.config)
        text = self.config.text_config
        for index, layer_type in enumerate(text.layer_types):
            if layer_type == "linear_attention":
                conv = torch.randn(1, 6144, 4, dtype=torch.bfloat16)
                recurrent = torch.randn(1, 16, 128, 128, dtype=torch.float32)
                cache.update_conv_state(conv, index)
                cache.update_recurrent_state(recurrent, index)
            else:
                keys = torch.randn(1, 2, 7, 256, dtype=torch.bfloat16)
                values = torch.randn(1, 2, 7, 256, dtype=torch.bfloat16)
                cache.update(keys, values, index)

        cloned = clone_dynamic_cache(cache, self.config)
        assert_cache_storage_independent(cache, cloned)
        source_layers, source_bytes = describe_cache(cache)
        cloned_layers, cloned_bytes = describe_cache(cloned)
        self.assertEqual(source_bytes, cloned_bytes)
        self.assertEqual(source_layers, cloned_layers)
        self.assertEqual(len(source_layers), 24)


if __name__ == "__main__":
    unittest.main()
