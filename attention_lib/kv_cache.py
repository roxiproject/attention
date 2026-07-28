"""KV-cache-style incremental decoding.

The correctness bar: running a full forward pass over a sequence must give
the exact same output as running the sequence one token at a time while
appending each new token's key/value into a cache and reusing it.
"""
import numpy as np

from attention_lib.sdpa import scaled_dot_product_attention
from attention_lib.positional import apply_rope


class KVCache:
    """Stores accumulated key/value tensors for one attention layer.

    Shapes are (batch, num_heads, seq_len_so_far, d_head); seq_len_so_far
    grows by one (or more) each time `append` is called.
    """

    def __init__(self):
        self.k = None
        self.v = None

    @property
    def seq_len(self) -> int:
        return 0 if self.k is None else self.k.shape[2]

    def append(self, k_new: np.ndarray, v_new: np.ndarray) -> None:
        if self.k is None:
            self.k = k_new
            self.v = v_new
        else:
            self.k = np.concatenate([self.k, k_new], axis=2)
            self.v = np.concatenate([self.v, v_new], axis=2)


def project_qkv(x: np.ndarray, w_q, w_k, w_v, num_heads: int):
    """Project x (batch, seq, d_model) to per-head q, k, v of shape
    (batch, num_heads, seq, d_head).
    """
    batch, seq, d_model = x.shape
    d_head = d_model // num_heads
    q = (x @ w_q).reshape(batch, seq, num_heads, d_head).transpose(0, 2, 1, 3)
    k = (x @ w_k).reshape(batch, seq, num_heads, d_head).transpose(0, 2, 1, 3)
    v = (x @ w_v).reshape(batch, seq, num_heads, d_head).transpose(0, 2, 1, 3)
    return q, k, v


def full_forward(x: np.ndarray, w_q, w_k, w_v, w_o, num_heads: int, use_rope: bool = False):
    """Standard full-sequence causal forward pass (no cache)."""
    q, k, v = project_qkv(x, w_q, w_k, w_v, num_heads)
    if use_rope:
        q = apply_rope(q)
        k = apply_rope(k)
    out, _ = scaled_dot_product_attention(q, k, v, causal=True)
    batch, num_heads_, seq, d_head = out.shape
    out = out.transpose(0, 2, 1, 3).reshape(batch, seq, num_heads_ * d_head)
    return out @ w_o


def incremental_forward_step(x_t: np.ndarray, cache: KVCache, w_q, w_k, w_v, w_o,
                              num_heads: int, use_rope: bool = False):
    """Process a single new token x_t (batch, 1, d_model), updating `cache`
    in place, and return that token's output (batch, 1, d_model).
    """
    q, k_new, v_new = project_qkv(x_t, w_q, w_k, w_v, num_heads)
    if use_rope:
        start_pos = cache.seq_len
        q = apply_rope(q, start_pos=start_pos)
        k_new = apply_rope(k_new, start_pos=start_pos)

    cache.append(k_new, v_new)

    out, _ = scaled_dot_product_attention(q, cache.k, cache.v, causal=True)
    batch, num_heads_, seq_q, d_head = out.shape
    out = out.transpose(0, 2, 1, 3).reshape(batch, seq_q, num_heads_ * d_head)
    return out @ w_o
