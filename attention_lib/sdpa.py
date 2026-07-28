"""Scaled dot-product attention."""
import numpy as np

from attention_lib.functional import softmax


def causal_mask(seq_len_q: int, seq_len_k: int) -> np.ndarray:
    """Boolean mask of shape (seq_len_q, seq_len_k), True where attention
    is allowed (key position <= query position, aligned to the end of the
    key sequence when seq_len_k > seq_len_q, as happens with a KV cache).
    """
    offset = seq_len_k - seq_len_q
    q_idx = np.arange(seq_len_q)[:, None]
    k_idx = np.arange(seq_len_k)[None, :]
    return (k_idx <= q_idx + offset)


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
    causal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Scaled dot-product attention.

    q: (..., seq_len_q, d_k)
    k: (..., seq_len_k, d_k)
    v: (..., seq_len_k, d_v)
    mask: optional boolean array broadcastable to (..., seq_len_q, seq_len_k),
          True = keep, False = mask out.
    causal: if True, apply a causal mask (query i can only see keys <= i).

    Returns (output, attn_weights) where output has shape (..., seq_len_q, d_v)
    and attn_weights has shape (..., seq_len_q, seq_len_k).
    """
    d_k = q.shape[-1]
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(d_k)

    if causal:
        seq_len_q, seq_len_k = q.shape[-2], k.shape[-2]
        cmask = causal_mask(seq_len_q, seq_len_k)
        scores = np.where(cmask, scores, -np.inf)

    if mask is not None:
        scores = np.where(mask, scores, -np.inf)

    attn_weights = softmax(scores, axis=-1)
    output = np.matmul(attn_weights, v)
    return output, attn_weights
