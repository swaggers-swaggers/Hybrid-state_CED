"""Explicit CPU-only math checks; no pretrained model, optimizer step, CUDA or data use."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ced_training.losses import readout_losses, normalized_mse, gate_loss
from ced_training.teacher import Teacher, prediction_rows


class NumericsTests(unittest.TestCase):
    def test_chunked_readout_gradient_matches_full_and_vocab_stays_frozen(self):
        torch.manual_seed(42)
        mapping = nn.Linear(4, 4, bias=False)
        vocab = nn.Linear(4, 7, bias=False).requires_grad_(False)
        h = torch.randn(9, 4)
        teacher = torch.randn(9, 7)
        labels = torch.arange(9) % 7
        ce, kd = readout_losses(vocab(mapping(h)), teacher, labels, 2.)
        (ce + kd).backward()
        expected = mapping.weight.grad.clone()
        mapping.zero_grad(set_to_none=True)
        for start in range(0, 9, 4):
            end = min(start + 4, 9)
            ce, kd = readout_losses(vocab(mapping(h[start:end])), teacher[start:end], labels[start:end], 2.)
            ((ce + kd) * ((end - start) / 9)).backward()
        torch.testing.assert_close(mapping.weight.grad, expected)
        self.assertGreater(expected.abs().sum().item(), 0)
        self.assertIsNone(vocab.weight.grad)

    def test_kd_direction_and_target_detachment(self):
        logits = torch.tensor([[2., -1., 0.]], requires_grad=True)
        reference = torch.tensor([[0., 1., 2.]], requires_grad=True)
        _, kd = readout_losses(logits, reference, torch.tensor([2]))
        p, q = reference.detach().softmax(-1), logits.softmax(-1)
        torch.testing.assert_close(kd, (p * (p.log() - q.log())).sum())
        kd.backward()
        self.assertIsNone(reference.grad)
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_projection_scaling_and_gate_loss(self):
        target = torch.tensor([[2., 2.]])
        predicted = torch.tensor([[1., 3.]], requires_grad=True)
        loss = normalized_mse(predicted, target, 4.)
        self.assertAlmostEqual(loss.item(), .25)
        loss.backward()
        self.assertGreater(predicted.grad.abs().sum().item(), 0)
        self.assertLess(gate_loss(torch.tensor([-5., 5.]), torch.tensor([0., 1.])).item(), .01)

    def test_oracle_state_control_preserves_prefix_and_held_gdn(self):
        from ced_training.evaluation import correct_state
        import copy
        layers = []
        for depth in range(1, 25):
            if depth in (4, 8, 12, 16, 20, 24):
                layer = SimpleNamespace(keys=torch.ones(1, 1, 3, 2), values=torch.ones(1, 1, 3, 2))
            else:
                layer = SimpleNamespace(conv_states=torch.ones(1, 2, 4), recurrent_states=torch.ones(1, 2, 2), has_previous_state=True)
            layers.append(layer)
        original = SimpleNamespace(layers=layers)
        shadow = copy.deepcopy(original)
        for depth in range(13, 25):
            layer = shadow.layers[depth - 1]
            if depth in (16, 20, 24):
                layer.keys[:, :, -1:, :].fill_(7.)
                layer.values[:, :, -1:, :].fill_(9.)
            else:
                layer.conv_states.fill_(3.)
                layer.recurrent_states.fill_(5.)
        held = copy.deepcopy(original)
        correct_state(None, held, shadow, kv_oracle=True, update_gdn=False)
        for depth in (16, 20, 24):
            torch.testing.assert_close(held.layers[depth-1].keys, shadow.layers[depth-1].keys)
            torch.testing.assert_close(held.layers[depth-1].keys[:, :, :-1], original.layers[depth-1].keys[:, :, :-1])
        torch.testing.assert_close(held.layers[12].conv_states, original.layers[12].conv_states)
        complete = copy.deepcopy(original)
        correct_state(None, complete, shadow, kv_oracle=True, update_gdn=True)
        self.assertNotEqual(complete.layers[12].conv_states.data_ptr(), shadow.layers[12].conv_states.data_ptr())
        torch.testing.assert_close(complete.layers[12].conv_states, shadow.layers[12].conv_states)

    def test_completion_mask_excludes_prompt_and_post_eos_padding(self):
        tokens = torch.tensor([[1, 2, 3, 10, 99, 0, 0], [4, 5, 6, 20, 21, 22, 23]])
        hidden = torch.arange(14).reshape(2, 7, 1)
        values, labels = prediction_rows({"h12": hidden}, tokens, start_position=2, eos_token_ids=[99])
        self.assertEqual(labels.tolist(), [10, 99, 20, 21, 22, 23])
        self.assertEqual(values["h12"].flatten().tolist(), [2, 3, 9, 10, 11, 12])
        _, first_eos = prediction_rows({"h12": hidden[:1]}, torch.tensor([[1, 2, 3, 99, 99, 99, 99]]),
                                      start_position=2, eos_token_ids=[99])
        self.assertEqual(first_eos.tolist(), [99])

    def test_warm_start_loads_main_modules_but_not_confidence(self):
        import tempfile
        from ced_training.engine import load_warm_start
        from ced_training.protocol import SEMANTICS
        config = {"seed": 7, "distilled_data_path": "/generated"}
        data = SimpleNamespace(distilled={"status":"COMPLETE"}, manifest_hash="raw-hash", splits={"modules": [1]},
                               manifest={"model_revision":"rev", "weights_etag_sha256":"model-hash"})
        runner = SimpleNamespace(readout_map=nn.Linear(2, 2), kv_projectors=nn.Linear(2, 2), confidence_head=nn.Linear(2, 1))
        gate = {k:v.detach().clone() for k,v in runner.confidence_head.state_dict().items()}
        state = {"schema_version":1, "semantics":SEMANTICS, "stage":"modules", "config":{"seed":7},
                 "profile":"pilot", "data_manifest_sha256":"raw-hash", "split_indices":data.splits,
                 "model_revision":"rev", "model_weights_sha256":"model-hash", "scales":{},
                 "modules":{name:{k:torch.full_like(v, 5.) for k,v in getattr(runner,name).state_dict().items()}
                            for name in ("readout_map","kv_projectors","confidence_head")}}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"modules.pt"
            torch.save(state,path)
            load_warm_start(path,runner,config,"pilot",data)
            torch.testing.assert_close(runner.readout_map.weight, torch.full_like(runner.readout_map.weight, 5.))
            for k,v in gate.items(): torch.testing.assert_close(runner.confidence_head.state_dict()[k],v)
            state["model_weights_sha256"]="wrong"
            torch.save(state,path)
            with self.assertRaises(ValueError): load_warm_start(path,runner,config,"pilot",data)

    def test_incremental_mask_has_no_off_by_one_or_extra_labels(self):
        from ced_training.incremental import masked_prediction_rows
        tokens = torch.tensor([[10,11,12,13],[20,21,22,23]])
        h = torch.arange(8).reshape(2,4,1)
        values, labels = masked_prediction_rows({"h12":h},tokens,torch.tensor([[0,1,1,0],[1,0,0,0]]))
        self.assertEqual(labels.tolist(),[12,13,21])
        self.assertEqual(values["h12"].flatten().tolist(),[1,2,4])
        with self.assertRaises(ValueError):
            masked_prediction_rows({"h12":h},tokens,torch.ones_like(tokens))

    def test_next_token_alignment_never_crosses_windows(self):
        tokens = torch.tensor([[10, 11, 12], [20, 21, 22]])
        h = torch.arange(6).reshape(2, 3, 1)
        values, labels = prediction_rows({"h12": h}, tokens)
        self.assertEqual(labels.tolist(), [11, 12, 21, 22])
        self.assertEqual(values["h12"].flatten().tolist(), [0, 1, 3, 4])

    def test_capture_uses_complete_block_and_actual_normalized_kv(self):
        class Block(nn.Module):
            def __init__(self, attention=False):
                super().__init__()
                self.input_layernorm = nn.LayerNorm(4)
                if attention:
                    self.self_attn = nn.Module()
                    self.self_attn.k_proj = nn.Linear(4, 2, bias=False)
                    self.self_attn.v_proj = nn.Linear(4, 2, bias=False)
            def forward(self, x):
                if hasattr(self, "self_attn"):
                    self.self_attn.k_proj(self.input_layernorm(x))
                    self.self_attn.v_proj(self.input_layernorm(x))
                return x + torch.tensor([1., 2., 3., 4.])
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Block(i in (15, 19, 23)) for i in range(24)])
                self.inputs = {}
            def forward(self, input_ids, use_cache):
                x = input_ids.float().unsqueeze(-1).expand(-1, -1, 4)
                for i, layer in enumerate(self.layers):
                    self.inputs[i] = x
                    x = layer(x)
                return SimpleNamespace(last_hidden_state=x)
        lm = Model()
        values = Teacher(SimpleNamespace(lm=lm)).capture(torch.tensor([[1, 2, 3]]))
        torch.testing.assert_close(values["h12"], lm.inputs[12])
        for depth in (16, 20, 24):
            layer = lm.layers[depth - 1]
            expected = layer.self_attn.k_proj(layer.input_layernorm(lm.inputs[depth - 1]))
            torch.testing.assert_close(values[f"k{depth}"], expected)
            wrong = layer.self_attn.k_proj(lm.inputs[depth - 1])
            self.assertFalse(torch.allclose(values[f"k{depth}"], wrong))
        self.assertFalse(values["h12"].requires_grad)
        self.assertEqual(sum(len(m._forward_hooks) for m in lm.modules()), 0)


if __name__ == "__main__":
    unittest.main()
