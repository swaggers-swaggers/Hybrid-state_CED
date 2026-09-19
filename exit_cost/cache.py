"""Independent cache copies for matched decode measurements."""
from typing import Any
from transformers.cache_utils import DynamicCache, LinearAttentionCacheLayerMixin


def clone_dynamic_cache(source: DynamicCache, config: Any) -> DynamicCache:
    """Reconstruct a fresh cache using only public cache update methods.

    Tensor storage is cloned so a measurement cannot mutate its saved prefix.
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

