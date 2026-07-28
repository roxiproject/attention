import numpy as np
import pytest

from attention_lib.sdpa import scaled_dot_product_attention, causal_mask


def naive_sdpa_loop(q, k, v):
    """Reference implementation using explicit Python loops, for a single
    (seq_len_q, d_k) / (seq_len_k, d_k) / (seq_len_k, d_v) triple (no batch
    or head dims). Used to independently verify the vectorized version.
    """
    seq_len_q, d_k = q.shape
    seq_len_k = k.shape[0]
    d_v = v.shape[1]
    out = np.zeros((seq_len_q, d_v))
    weights = np.zeros((seq_len_q, seq_len_k))
    for i in range(seq_len_q):
        scores = np.zeros(seq_len_k)
        for j in range(seq_len_k):
            s = 0.0
            for t in range(d_k):
                s += q[i, t] * k[j, t]
            scores[j] = s / np.sqrt(d_k)
        m = np.max(scores)
        exp_scores = np.exp(scores - m)
        probs = exp_scores / np.sum(exp_scores)
        weights[i] = probs
        for t in range(d_v):
            acc = 0.0
            for j in range(seq_len_k):
                acc += probs[j] * v[j, t]
            out[i, t] = acc
    return out, weights


@pytest.fixture
def small_qkv():
    rng = np.random.default_rng(0)
    q = rng.standard_normal((4, 8))
    k = rng.standard_normal((5, 8))
    v = rng.standard_normal((5, 6))
    return q, k, v


def test_sdpa_matches_naive_loop(small_qkv):
    q, k, v = small_qkv
    out, weights = scaled_dot_product_attention(q, k, v)
    ref_out, ref_weights = naive_sdpa_loop(q, k, v)
    assert np.allclose(out, ref_out, atol=1e-8)
    assert np.allclose(weights, ref_weights, atol=1e-8)


def test_sdpa_output_shape(small_qkv):
    q, k, v = small_qkv
    out, weights = scaled_dot_product_attention(q, k, v)
    assert out.shape == (4, 6)
    assert weights.shape == (4, 5)


def test_sdpa_weights_sum_to_one(small_qkv):
    q, k, v = small_qkv
    _, weights = scaled_dot_product_attention(q, k, v)
    assert np.allclose(np.sum(weights, axis=-1), 1.0)


def test_sdpa_batched():
    rng = np.random.default_rng(1)
    q = rng.standard_normal((3, 4, 8))
    k = rng.standard_normal((3, 5, 8))
    v = rng.standard_normal((3, 5, 6))
    out, weights = scaled_dot_product_attention(q, k, v)
    assert out.shape == (3, 4, 6)
    for b in range(3):
        ref_out, ref_weights = naive_sdpa_loop(q[b], k[b], v[b])
        assert np.allclose(out[b], ref_out, atol=1e-8)
        assert np.allclose(weights[b], ref_weights, atol=1e-8)


def test_causal_mask_shape():
    m = causal_mask(4, 4)
    assert m.shape == (4, 4)


def test_causal_mask_lower_triangular():
    m = causal_mask(4, 4)
    expected = np.tril(np.ones((4, 4), dtype=bool))
    assert np.array_equal(m, expected)


def test_sdpa_causal_masks_future_positions():
    rng = np.random.default_rng(2)
    seq = 5
    q = rng.standard_normal((seq, 8))
    k = rng.standard_normal((seq, 8))
    v = rng.standard_normal((seq, 6))
    _, weights = scaled_dot_product_attention(q, k, v, causal=True)
    for i in range(seq):
        for j in range(seq):
            if j > i:
                assert weights[i, j] == 0.0


def test_sdpa_causal_weights_sum_to_one():
    rng = np.random.default_rng(3)
    q = rng.standard_normal((5, 8))
    k = rng.standard_normal((5, 8))
    v = rng.standard_normal((5, 6))
    _, weights = scaled_dot_product_attention(q, k, v, causal=True)
    assert np.allclose(np.sum(weights, axis=-1), 1.0)


def test_sdpa_causal_first_position_attends_only_to_self():
    rng = np.random.default_rng(4)
    q = rng.standard_normal((5, 8))
    k = rng.standard_normal((5, 8))
    v = rng.standard_normal((5, 6))
    _, weights = scaled_dot_product_attention(q, k, v, causal=True)
    assert np.isclose(weights[0, 0], 1.0)


def test_sdpa_explicit_mask_matches_causal():
    rng = np.random.default_rng(5)
    seq = 6
    q = rng.standard_normal((seq, 8))
    k = rng.standard_normal((seq, 8))
    v = rng.standard_normal((seq, 6))
    out_causal, w_causal = scaled_dot_product_attention(q, k, v, causal=True)
    mask = np.tril(np.ones((seq, seq), dtype=bool))
    out_mask, w_mask = scaled_dot_product_attention(q, k, v, mask=mask)
    assert np.allclose(out_causal, out_mask)
    assert np.allclose(w_causal, w_mask)


def test_sdpa_full_attention_no_mask_uses_all_keys():
    rng = np.random.default_rng(6)
    q = rng.standard_normal((3, 4))
    k = rng.standard_normal((7, 4))
    v = rng.standard_normal((7, 2))
    _, weights = scaled_dot_product_attention(q, k, v)
    assert np.all(weights > 0)


def test_sdpa_identical_qk_gives_peaked_weights():
    # When q == k rows are far apart in direction, attention should favor
    # matching indices along the diagonal for an orthogonal-ish basis.
    d = 16
    seq = 4
    q = np.eye(seq, d) * 10.0
    k = np.eye(seq, d) * 10.0
    v = np.arange(seq)[:, None].astype(float) * np.ones((seq, 3))
    out, weights = scaled_dot_product_attention(q, k, v)
    assert np.allclose(np.argmax(weights, axis=-1), np.arange(seq))
