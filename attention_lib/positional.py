"""Positional encodings: rotary position embeddings (RoPE) and a simple
relative positional bias.
"""
import numpy as np


def rope_frequencies(d_head: int, base: float = 10000.0) -> np.ndarray:
    """Inverse frequencies theta_i = base^(-2i/d) for i in [0, d/2)."""
    assert d_head % 2 == 0, "RoPE requires an even head dimension"
    i = np.arange(0, d_head, 2, dtype=np.float64)
    return 1.0 / (base ** (i / d_head))


def rope_angles(seq_len: int, d_head: int, base: float = 10000.0) -> np.ndarray:
    """Angles of shape (seq_len, d_head // 2): position * inverse_frequency."""
    freqs = rope_frequencies(d_head, base)
    positions = np.arange(seq_len, dtype=np.float64)
    return np.outer(positions, freqs)


def apply_rope(x: np.ndarray, start_pos: int = 0, base: float = 10000.0) -> np.ndarray:
    """Apply rotary position embeddings to x.

    x: (..., seq_len, d_head), d_head even. Pairs (x[..., 2i], x[..., 2i+1])
    are treated as 2D vectors and rotated by angle = position * theta_i,
    where position runs from start_pos to start_pos + seq_len - 1 (so an
    incrementally-decoded token at absolute position p gets the same
    rotation as if it had been part of a full forward pass up to p).
    """
    *batch_shape, seq_len, d_head = x.shape
    positions = np.arange(start_pos, start_pos + seq_len, dtype=np.float64)
    angles = np.outer(positions, rope_frequencies(d_head, base))  # (seq_len, d_head // 2)

    cos = np.cos(angles)  # (seq_len, d_head // 2)
    sin = np.sin(angles)

    x_pairs = x.reshape(*batch_shape, seq_len, d_head // 2, 2)
    x1 = x_pairs[..., 0]
    x2 = x_pairs[..., 1]

    # broadcast cos/sin over leading batch dims
    cos_b = cos.reshape((1,) * len(batch_shape) + cos.shape)
    sin_b = sin.reshape((1,) * len(batch_shape) + sin.shape)

    rot1 = x1 * cos_b - x2 * sin_b
    rot2 = x1 * sin_b + x2 * cos_b

    rotated = np.stack([rot1, rot2], axis=-1)
    return rotated.reshape(*batch_shape, seq_len, d_head)


def relative_position_bias(seq_len_q: int, seq_len_k: int, num_buckets: int = 32,
                            max_distance: int = 128) -> np.ndarray:
    """A simple T5-style relative position bucket matrix (integers), shape
    (seq_len_q, seq_len_k), suitable for indexing into a learned bias table.
    Distances are clipped and log-bucketed for far positions.
    """
    q_idx = np.arange(seq_len_q)[:, None]
    k_idx = np.arange(seq_len_k)[None, :]
    relative_position = k_idx - q_idx  # negative = key is before query

    num_buckets_half = num_buckets // 2
    ret = np.where(relative_position > 0, num_buckets_half, 0)
    n = np.abs(relative_position)

    max_exact = num_buckets_half // 2
    is_small = n < max_exact

    val_if_large = max_exact + (
        np.log(np.maximum(n, 1).astype(np.float64) / max_exact)
        / np.log(max_distance / max_exact)
        * (num_buckets_half - max_exact)
    ).astype(np.int64)
    val_if_large = np.minimum(val_if_large, num_buckets_half - 1)

    bucket = np.where(is_small, n, val_if_large)
    return ret + bucket
