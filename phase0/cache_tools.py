from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import DynamicCache, LinearAttentionCacheLayerMixin

from .state_map import tensor_metadata


def clone_dynamic_cache(source: DynamicCache, config: Any) -> DynamicCache:
    """Reconstruct a fresh cache using only public cache update methods.

    This is the core Phase 0 extraction -> reinjection operation. Tensor storage is
    cloned so the teacher path and reinjected path cannot mutate one another.
    """
    target = DynamicCache(config=config)
    if len(target.layers) != len(source.layers):
        raise ValueError(f"Cache layer count mismatch: {len(source.layers)} != {len(target.layers)}")

    for index, source_layer in enumerate(source.layers):
        if isinstance(source_layer, LinearAttentionCacheLayerMixin):
            if source_layer.conv_states is not None:
                target.update_conv_state(source_layer.conv_states.detach().clone(), index)
            if source_layer.recurrent_states is not None:
                target.update_recurrent_state(source_layer.recurrent_states.detach().clone(), index)
            target.layers[index].has_previous_state = source_layer.has_previous_state
        else:
            keys = getattr(source_layer, "keys", None)
            values = getattr(source_layer, "values", None)
            if keys is not None and values is not None:
                target.update(keys.detach().clone(), values.detach().clone(), index)
    return target


def describe_cache(cache: DynamicCache) -> tuple[list[dict[str, Any]], int]:
    descriptions: list[dict[str, Any]] = []
    total_bytes = 0
    for index, layer in enumerate(cache.layers):
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            states = {
                "conv": tensor_metadata(layer.conv_states),
                "recurrent": tensor_metadata(layer.recurrent_states),
            }
            kind = "gdn"
            initialized = bool(layer.has_previous_state)
        else:
            states = {
                "keys": tensor_metadata(getattr(layer, "keys", None)),
                "values": tensor_metadata(getattr(layer, "values", None)),
            }
            kind = "attention_kv"
            initialized = bool(getattr(layer, "is_initialized", False))
        layer_bytes = sum(state["bytes"] for state in states.values() if state is not None)
        total_bytes += layer_bytes
        descriptions.append(
            {
                "layer": index,
                "kind": kind,
                "initialized": initialized,
                "states": states,
                "bytes": layer_bytes,
            }
        )
    return descriptions, total_bytes


def compare_tensors(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape:
        return {"shape_equal": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    l = left.detach().float()
    r = right.detach().float()
    difference = (l - r).abs()
    flat_l = l.reshape(-1)
    flat_r = r.reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(flat_l, flat_r, dim=0).item()
    return {
        "shape_equal": True,
        "max_abs": difference.max().item() if difference.numel() else 0.0,
        "mean_abs": difference.mean().item() if difference.numel() else 0.0,
        "cosine": cosine,
        "top1_equal": bool(left[..., -1, :].argmax(-1).eq(right[..., -1, :].argmax(-1)).all())
        if left.ndim >= 2
        else None,
    }


def compare_logits(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    metrics = compare_tensors(left, right)
    left_log_prob = torch.nn.functional.log_softmax(left.detach().float(), dim=-1)
    right_log_prob = torch.nn.functional.log_softmax(right.detach().float(), dim=-1)
    left_prob = left_log_prob.exp()
    metrics["kl_left_to_right"] = (left_prob * (left_log_prob - right_log_prob)).sum(dim=-1).mean().item()
    difference_norm = (left.detach().float() - right.detach().float()).norm()
    metrics["relative_l2"] = (difference_norm / left.detach().float().norm().clamp_min(1e-12)).item()
    return metrics


def assert_cache_storage_independent(left: DynamicCache, right: DynamicCache) -> None:
    for left_layer, right_layer in zip(left.layers, right.layers, strict=True):
        names = ("conv_states", "recurrent_states") if isinstance(left_layer, LinearAttentionCacheLayerMixin) else ("keys", "values")
        for name in names:
            left_tensor = getattr(left_layer, name, None)
            right_tensor = getattr(right_layer, name, None)
            if left_tensor is not None and right_tensor is not None and left_tensor.data_ptr() == right_tensor.data_ptr():
                raise AssertionError(f"Cache tensor {name} shares storage after reinjection")
