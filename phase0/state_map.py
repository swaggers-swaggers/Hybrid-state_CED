from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


DTYPE_BYTES = {
    torch.float64: 8,
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int64: 8,
    torch.int32: 4,
    torch.int16: 2,
    torch.int8: 1,
    torch.uint8: 1,
    torch.bool: 1,
}


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def tensor_metadata(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "device": str(tensor.device),
        "bytes": tensor_nbytes(tensor),
        "requires_grad": tensor.requires_grad,
    }


def _dtype_size(dtype_name: str) -> int:
    normalized = dtype_name.removeprefix("torch.")
    for dtype, size in DTYPE_BYTES.items():
        if str(dtype).removeprefix("torch.") == normalized:
            return size
    raise ValueError(f"Unsupported dtype for static estimate: {dtype_name}")


@dataclass(frozen=True)
class ArchitectureSummary:
    num_hidden_layers: int
    hidden_size: int
    linear_attention_layers: int
    full_attention_layers: int
    full_attention_indices: tuple[int, ...]


def build_static_state_map(text_config: Any, batch_size: int = 1, dtype_name: str = "bfloat16") -> dict[str, Any]:
    """Build the state map implied by the official model config.

    Static estimates are intentionally kept separate from runtime observations.
    """
    dtype_bytes = _dtype_size(dtype_name)
    layers: list[dict[str, Any]] = []
    full_indices: list[int] = []
    linear_count = 0

    for index, layer_type in enumerate(text_config.layer_types):
        hidden_per_token = batch_size * text_config.hidden_size * dtype_bytes
        base: dict[str, Any] = {
            "layer": index,
            "type": layer_type,
            "hidden": {
                "shape_formula": [batch_size, "N", text_config.hidden_size],
                "bytes_formula": f"{hidden_per_token} * N",
                "growth": "sequence",
                "dependency": "previous layer residual stream",
            },
        }
        if layer_type == "full_attention":
            full_indices.append(index)
            kv_heads = text_config.num_key_value_heads
            head_dim = text_config.head_dim
            bytes_per_k_or_v_token = batch_size * kv_heads * head_dim * dtype_bytes
            base["cache"] = {
                "kind": "attention_kv",
                "raw_k_shape_formula": [batch_size, "N", kv_heads, head_dim],
                "knorm_k_shape_formula": [batch_size, "N", kv_heads, head_dim],
                "final_k_shape_formula": [batch_size, kv_heads, "N", head_dim],
                "v_shape_formula": [batch_size, kv_heads, "N", head_dim],
                "k_bytes_formula": f"{bytes_per_k_or_v_token} * N",
                "v_bytes_formula": f"{bytes_per_k_or_v_token} * N",
                "growth": "sequence",
                "dependency": "normalized hidden -> k/v projections; KNorm and RoPE for K; cache concatenation by position",
            }
        else:
            linear_count += 1
            channels = (
                2 * text_config.linear_num_key_heads * text_config.linear_key_head_dim
                + text_config.linear_num_value_heads * text_config.linear_value_head_dim
            )
            conv_shape = [batch_size, channels, text_config.linear_conv_kernel_dim]
            recurrent_shape = [
                batch_size,
                text_config.linear_num_value_heads,
                text_config.linear_key_head_dim,
                text_config.linear_value_head_dim,
            ]
            conv_bytes = batch_size * channels * text_config.linear_conv_kernel_dim * dtype_bytes
            # DynamicCache initializes its storage from the BF16 conv-state dtype;
            # a fp32 GDN compute result is cast into this cache storage on update.
            recurrent_bytes = (
                batch_size
                * text_config.linear_num_value_heads
                * text_config.linear_key_head_dim
                * text_config.linear_value_head_dim
                * dtype_bytes
            )
            base["cache"] = {
                "kind": "gdn",
                "conv_shape": conv_shape,
                "conv_bytes": conv_bytes,
                "recurrent_shape": recurrent_shape,
                "recurrent_bytes": recurrent_bytes,
                "growth": "static",
                "dependency": "last convolution window plus prefix-folded DeltaNet associative recurrence",
            }
        layers.append(base)

    return {
        "architecture": {
            "num_hidden_layers": len(text_config.layer_types),
            "hidden_size": text_config.hidden_size,
            "linear_attention_layers": linear_count,
            "full_attention_layers": len(full_indices),
            "full_attention_indices": full_indices,
            "conv_kernel_size": text_config.linear_conv_kernel_dim,
            "native_context": text_config.max_position_embeddings,
        },
        "assumptions": {"batch_size": batch_size, "activation_dtype": dtype_name},
        "layers": layers,
    }


def validate_architecture(static_map: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    architecture = static_map["architecture"]
    for key, expected_value in expected.items():
        actual = architecture.get(key)
        if actual != expected_value:
            errors.append(f"{key}: expected {expected_value!r}, got {actual!r}")
    return errors
