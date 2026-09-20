"""Online targets from the frozen full model; never persist full vocabulary logits."""
from __future__ import annotations
import torch
from exit_cost.runtime import rotate_key
from .protocol import TARGETS


class Teacher:
    def __init__(self, runner):
        self.runner = runner

    @torch.no_grad()
    def capture(self, tokens, verify=False):
        captured, handles = {}, []
        def remember(name):
            def hook(module, args, output):
                captured[name] = output.detach()
            return hook
        # Actual linear outputs automatically use self_attn's post-input_layernorm input.
        handles.append(self.runner.lm.layers[11].register_forward_hook(remember("h12")))
        for depth in TARGETS:
            attn = self.runner.lm.layers[depth - 1].self_attn
            handles.append(attn.k_proj.register_forward_hook(remember(f"k{depth}")))
            handles.append(attn.v_proj.register_forward_hook(remember(f"v{depth}")))
        try:
            result = self.runner.lm(input_ids=tokens, use_cache=verify)
        finally:
            for handle in handles:
                handle.remove()
        captured["final"] = result.last_hidden_state.detach()  # already final-normalized
        if verify:
            batch, length = tokens.shape
            positions = torch.arange(length, device=tokens.device).view(1, 1, -1).expand(3, batch, -1)
            rope = self.runner.lm.rotary_emb(captured["h12"], positions)
            for depth in TARGETS:
                attn = self.runner.lm.layers[depth - 1].self_attn
                shape = (batch, length, -1, attn.head_dim)
                k = rotate_key(attn.k_norm(captured[f"k{depth}"].view(shape)).transpose(1, 2), *rope)
                v = captured[f"v{depth}"].view(shape).transpose(1, 2)
                layer = result.past_key_values.layers[depth - 1]
                torch.testing.assert_close(k, layer.keys, rtol=0, atol=0)
                torch.testing.assert_close(v, layer.values, rtol=0, atol=0)
        return captured


def prediction_rows(captured, tokens, start_position=0, eos_token_ids=None):
    # h[t] predicts token[t+1]. In generated data, supervise completion labels only.
    labels = tokens[:, start_position + 1:]
    selected = {name: value[:, start_position:-1] for name, value in captured.items()}
    valid = torch.ones_like(labels, dtype=torch.bool)
    if eos_token_ids:
        eos = torch.zeros_like(labels, dtype=torch.bool)
        for token_id in eos_token_ids:
            eos |= labels == token_id
        # Include the first EOS, exclude every position after it (including EOS padding).
        valid = (eos.long().cumsum(-1) - eos.long()) == 0
    return ({name: value[valid] for name, value in selected.items()}, labels[valid])
