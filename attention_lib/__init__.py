"""attention_lib: numpy-only implementations of attention mechanisms."""

from attention_lib.functional import softmax
from attention_lib.sdpa import scaled_dot_product_attention, causal_mask
from attention_lib.multihead import (
    MultiHeadAttention,
    split_heads,
    combine_heads,
    grouped_query_attention,
)
from attention_lib.positional import apply_rope, rope_frequencies, relative_position_bias
from attention_lib.kv_cache import KVCache, full_forward, incremental_forward_step

__version__ = "0.1.0"

__all__ = [
    "softmax",
    "scaled_dot_product_attention",
    "causal_mask",
    "MultiHeadAttention",
    "split_heads",
    "combine_heads",
    "grouped_query_attention",
    "apply_rope",
    "rope_frequencies",
    "relative_position_bias",
    "KVCache",
    "full_forward",
    "incremental_forward_step",
]
