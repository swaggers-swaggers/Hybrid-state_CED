from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

TARGET_SEMANTICS = "self_attn_input_post_layernorm_raw_kv_v2"


def require_corrected_checkpoint(checkpoint: dict) -> None:
    if checkpoint.get("target_semantics") != TARGET_SEMANTICS:
        raise ValueError("Legacy or unknown target semantics: retrain Phase 1 with post-input_layernorm targets")


class QwenFeatureCapture:
    """Capture source residuals and the target attention's actual normalized input."""

    def __init__(self, model: Any, source_layers: Iterable[int], target_layer: int) -> None:
        self.model = model
        self.source_layers = tuple(source_layers)
        self.target_layer = target_layer
        self.sources: dict[int, torch.Tensor] = {}
        self.target_input: torch.Tensor | None = None
        self.handles: list[Any] = []
        self.cache_verification: dict[str, Any] | None = None

    def _source_hook(self, index: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            self.sources[index] = tensor.detach()

        return hook

    def _target_hook(self, _module: torch.nn.Module, inputs: tuple[Any, ...], kwargs: dict) -> None:
        hidden = inputs[0] if inputs else kwargs.get("hidden_states")
        if not torch.is_tensor(hidden):
            raise RuntimeError("Target self-attention hidden state was not captured")
        self.target_input = hidden.detach()

    def __enter__(self) -> "QwenFeatureCapture":
        layers = self.model.model.language_model.layers
        for index in self.source_layers:
            self.handles.append(layers[index].register_forward_hook(self._source_hook(index)))
        self.handles.append(layers[self.target_layer].self_attn.register_forward_pre_hook(self._target_hook, with_kwargs=True))
        return self

    def __exit__(self, *_args: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @torch.no_grad()
    def capture(self, input_ids: torch.Tensor) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        self.sources = {}
        self.target_input = None
        # First batch verifies the captured target against the real inference cache.
        verify = self.cache_verification is None
        output = self.model.model.language_model(input_ids=input_ids, use_cache=verify)
        if set(self.sources) != set(self.source_layers) or self.target_input is None:
            raise RuntimeError("Not all requested teacher features were captured")
        if verify:
            lm = self.model.model.language_model
            attention = lm.layers[self.target_layer].self_attn
            raw_k, raw_v = teacher_raw_kv(attention, self.target_input)
            _, key = normalized_qk(attention, self.target_input, raw_k, lm.rotary_emb)
            b, n, _ = raw_v.shape
            value = raw_v.reshape(b, n, -1, attention.head_dim).transpose(1, 2)
            cache = output.past_key_values.layers[self.target_layer]
            exact = torch.equal(key, cache.keys) and torch.equal(value, cache.values)
            self.cache_verification = {"layer": self.target_layer, "tokens": n,
                "batch_size": b, "exact_equal": exact,
                "key_max_abs": float((key-cache.keys).abs().max()),
                "value_max_abs": float((value-cache.values).abs().max())}
            if not exact:
                raise AssertionError(f"Captured target differs from real KV cache: {self.cache_verification}")
        return dict(self.sources), self.target_input


def teacher_raw_kv(attention: torch.nn.Module, target_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return attention.k_proj(target_hidden), attention.v_proj(target_hidden)


def normalized_qk(
    attention: torch.nn.Module,
    target_hidden: torch.Tensor,
    raw_k: torch.Tensor,
    rotary_emb: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, _ = target_hidden.shape
    hidden_shape = (batch_size, sequence_length, -1, attention.head_dim)
    query, _gate = torch.chunk(
        attention.q_proj(target_hidden).view(batch_size, sequence_length, -1, attention.head_dim * 2),
        2,
        dim=-1,
    )
    query = attention.q_norm(query.view(hidden_shape)).transpose(1, 2)
    key = attention.k_norm(raw_k.view(hidden_shape)).transpose(1, 2)
    position_ids = torch.arange(sequence_length, device=target_hidden.device).unsqueeze(0).expand(batch_size, -1)
    cos, sin = rotary_emb(target_hidden, position_ids)
    return apply_rotary_pos_emb(query, key, cos, sin)


def attention_core_output(
    attention: torch.nn.Module,
    rotary_emb: torch.nn.Module,
    target_hidden: torch.Tensor,
    raw_k: torch.Tensor,
    raw_v: torch.Tensor,
    teacher_query: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if teacher_query is None:
        query, key = normalized_qk(attention, target_hidden, raw_k, rotary_emb)
    else:
        _unused_query, key = normalized_qk(attention, target_hidden, raw_k, rotary_emb)
        query = teacher_query
    batch_size, sequence_length, _ = raw_v.shape
    value = raw_v.view(batch_size, sequence_length, -1, attention.head_dim).transpose(1, 2)
    groups = attention.config.num_attention_heads // attention.config.num_key_value_heads
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=True,
        dropout_p=0.0,
        scale=attention.scaling,
    )
    return output, query
