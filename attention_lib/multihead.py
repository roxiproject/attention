"""Multi-head attention (and grouped-query / multi-query attention)."""
import numpy as np

from attention_lib.sdpa import scaled_dot_product_attention


def split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    """(batch, seq, d_model) -> (batch, num_heads, seq, d_model // num_heads)."""
    batch, seq, d_model = x.shape
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    d_head = d_model // num_heads
    x = x.reshape(batch, seq, num_heads, d_head)
    return np.transpose(x, (0, 2, 1, 3))


def combine_heads(x: np.ndarray) -> np.ndarray:
    """(batch, num_heads, seq, d_head) -> (batch, seq, num_heads * d_head)."""
    batch, num_heads, seq, d_head = x.shape
    x = np.transpose(x, (0, 2, 1, 3))
    return x.reshape(batch, seq, num_heads * d_head)


class MultiHeadAttention:
    """Standard multi-head attention with learned (randomly initialized)
    projection matrices, implemented from scratch with numpy.
    """

    def __init__(self, d_model: int, num_heads: int, seed: int | None = None):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d_model)
        self.w_q = rng.standard_normal((d_model, d_model)) * scale
        self.w_k = rng.standard_normal((d_model, d_model)) * scale
        self.w_v = rng.standard_normal((d_model, d_model)) * scale
        self.w_o = rng.standard_normal((d_model, d_model)) * scale

    def __call__(self, x: np.ndarray, causal: bool = False) -> np.ndarray:
        """x: (batch, seq, d_model) -> (batch, seq, d_model)."""
        q = x @ self.w_q
        k = x @ self.w_k
        v = x @ self.w_v

        qh = split_heads(q, self.num_heads)
        kh = split_heads(k, self.num_heads)
        vh = split_heads(v, self.num_heads)

        out, _ = scaled_dot_product_attention(qh, kh, vh, causal=causal)
        combined = combine_heads(out)
        return combined @ self.w_o


def grouped_query_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    num_query_heads: int,
    num_kv_heads: int,
    causal: bool = False,
) -> np.ndarray:
    """Grouped-query / multi-query attention.

    q: (batch, seq_q, num_query_heads, d_head)
    k, v: (batch, seq_k, num_kv_heads, d_head)

    num_query_heads must be a multiple of num_kv_heads. Each group of
    (num_query_heads // num_kv_heads) query heads shares one KV head.
    Multi-query attention is the special case num_kv_heads == 1.
    """
    assert num_query_heads % num_kv_heads == 0
    group_size = num_query_heads // num_kv_heads

    batch, seq_q, _, d_head = q.shape
    seq_k = k.shape[1]

    q_bh = np.transpose(q, (0, 2, 1, 3))  # (batch, num_query_heads, seq_q, d_head)
    k_bh = np.transpose(k, (0, 2, 1, 3))  # (batch, num_kv_heads, seq_k, d_head)
    v_bh = np.transpose(v, (0, 2, 1, 3))

    # Repeat each kv head group_size times so it lines up with query heads.
    k_expanded = np.repeat(k_bh, group_size, axis=1)  # (batch, num_query_heads, seq_k, d_head)
    v_expanded = np.repeat(v_bh, group_size, axis=1)

    out, _ = scaled_dot_product_attention(q_bh, k_expanded, v_expanded, causal=causal)
    # out: (batch, num_query_heads, seq_q, d_head) -> (batch, seq_q, num_query_heads, d_head)
    return np.transpose(out, (0, 2, 1, 3))
