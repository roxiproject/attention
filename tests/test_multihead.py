import numpy as np
import pytest

from attention_lib.multihead import (
    MultiHeadAttention,
    split_heads,
    combine_heads,
    grouped_query_attention,
)


def test_split_heads_shape():
    x = np.random.randn(2, 5, 8)
    out = split_heads(x, num_heads=4)
    assert out.shape == (2, 4, 5, 2)


def test_combine_heads_shape():
    x = np.random.randn(2, 4, 5, 2)
    out = combine_heads(x)
    assert out.shape == (2, 5, 8)


def test_split_then_combine_is_identity():
    x = np.random.randn(3, 6, 12)
    out = combine_heads(split_heads(x, num_heads=3))
    assert np.allclose(out, x)


def test_split_heads_preserves_values():
    # Explicitly check that head h, position s, dim d maps back to the
    # correct slice of the original last axis.
    x = np.arange(2 * 4 * 8).reshape(2, 4, 8).astype(float)
    heads = split_heads(x, num_heads=2)
    d_head = 4
    for b in range(2):
        for s in range(4):
            for h in range(2):
                expected = x[b, s, h * d_head:(h + 1) * d_head]
                assert np.allclose(heads[b, h, s], expected)


def naive_mha_loop(x, w_q, w_k, w_v, w_o, num_heads):
    """Fully independent reference: explicit per-head, per-position loops
    with plain Python matmuls via nested sums, computed differently from
    the vectorized split/concat implementation.
    """
    batch, seq, d_model = x.shape
    d_head = d_model // num_heads

    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    output = np.zeros((batch, seq, d_model))
    for b in range(batch):
        for h in range(num_heads):
            lo, hi = h * d_head, (h + 1) * d_head
            qh = q[b, :, lo:hi]
            kh = k[b, :, lo:hi]
            vh = v[b, :, lo:hi]
            scores = np.zeros((seq, seq))
            for i in range(seq):
                for j in range(seq):
                    scores[i, j] = np.dot(qh[i], kh[j]) / np.sqrt(d_head)
            for i in range(seq):
                row = scores[i]
                m = np.max(row)
                e = np.exp(row - m)
                probs = e / np.sum(e)
                acc = np.zeros(d_head)
                for j in range(seq):
                    acc += probs[j] * vh[j]
                output[b, i, lo:hi] = acc
    return output @ w_o


def test_multihead_attention_matches_independent_loop_reference():
    d_model, num_heads = 8, 2
    mha = MultiHeadAttention(d_model, num_heads, seed=42)
    x = np.random.default_rng(42).standard_normal((2, 3, d_model))
    out = mha(x)
    ref = naive_mha_loop(x, mha.w_q, mha.w_k, mha.w_v, mha.w_o, num_heads)
    assert np.allclose(out, ref, atol=1e-8)


def test_multihead_attention_output_shape():
    mha = MultiHeadAttention(16, 4, seed=1)
    x = np.random.randn(3, 7, 16)
    out = mha(x)
    assert out.shape == (3, 7, 16)


def test_multihead_attention_causal_matches_loop_reference():
    d_model, num_heads = 8, 2
    mha = MultiHeadAttention(d_model, num_heads, seed=7)
    x = np.random.default_rng(7).standard_normal((1, 4, d_model))
    out = mha(x, causal=True)

    # independent causal reference
    q = x @ mha.w_q
    k = x @ mha.w_k
    v = x @ mha.w_v
    d_head = d_model // num_heads
    seq = x.shape[1]
    output = np.zeros((1, seq, d_model))
    for h in range(num_heads):
        lo, hi = h * d_head, (h + 1) * d_head
        qh, kh, vh = q[0, :, lo:hi], k[0, :, lo:hi], v[0, :, lo:hi]
        for i in range(seq):
            scores = np.array([np.dot(qh[i], kh[j]) / np.sqrt(d_head) if j <= i else -np.inf
                                for j in range(seq)])
            m = np.max(scores)
            e = np.exp(scores - m)
            probs = e / np.sum(e)
            acc = np.zeros(d_head)
            for j in range(seq):
                if j <= i:
                    acc += probs[j] * vh[j]
            output[0, i, lo:hi] = acc
    ref = output @ mha.w_o
    assert np.allclose(out, ref, atol=1e-8)


def test_multihead_attention_deterministic_with_seed():
    mha1 = MultiHeadAttention(8, 2, seed=99)
    mha2 = MultiHeadAttention(8, 2, seed=99)
    x = np.random.default_rng(0).standard_normal((1, 3, 8))
    assert np.allclose(mha1(x), mha2(x))


def test_multihead_attention_rejects_bad_head_count():
    with pytest.raises(AssertionError):
        MultiHeadAttention(10, 3)


# --- Grouped-query / multi-query attention ---

def test_gqa_output_shape():
    batch, seq_q, seq_k = 2, 4, 4
    num_q_heads, num_kv_heads, d_head = 8, 2, 4
    q = np.random.randn(batch, seq_q, num_q_heads, d_head)
    k = np.random.randn(batch, seq_k, num_kv_heads, d_head)
    v = np.random.randn(batch, seq_k, num_kv_heads, d_head)
    out = grouped_query_attention(q, k, v, num_q_heads, num_kv_heads)
    assert out.shape == (batch, seq_q, num_q_heads, d_head)


def test_gqa_equals_mha_when_num_kv_heads_equals_num_query_heads():
    from attention_lib.sdpa import scaled_dot_product_attention
    batch, seq, num_heads, d_head = 1, 5, 4, 3
    q = np.random.default_rng(0).standard_normal((batch, seq, num_heads, d_head))
    k = np.random.default_rng(1).standard_normal((batch, seq, num_heads, d_head))
    v = np.random.default_rng(2).standard_normal((batch, seq, num_heads, d_head))

    gqa_out = grouped_query_attention(q, k, v, num_heads, num_heads)

    q_bh = np.transpose(q, (0, 2, 1, 3))
    k_bh = np.transpose(k, (0, 2, 1, 3))
    v_bh = np.transpose(v, (0, 2, 1, 3))
    mha_out, _ = scaled_dot_product_attention(q_bh, k_bh, v_bh)
    mha_out = np.transpose(mha_out, (0, 2, 1, 3))

    assert np.allclose(gqa_out, mha_out)


def test_gqa_shared_kv_head_used_by_correct_query_group():
    # 4 query heads, 2 kv heads -> heads [0,1] share kv head 0, [2,3] share kv head 1.
    batch, seq, d_head = 1, 3, 4
    num_q_heads, num_kv_heads = 4, 2
    q = np.random.default_rng(3).standard_normal((batch, seq, num_q_heads, d_head))
    k = np.random.default_rng(4).standard_normal((batch, seq, num_kv_heads, d_head))
    v = np.random.default_rng(5).standard_normal((batch, seq, num_kv_heads, d_head))

    out = grouped_query_attention(q, k, v, num_q_heads, num_kv_heads)

    from attention_lib.sdpa import scaled_dot_product_attention
    for group, kv_idx in [((0, 1), 0), ((2, 3), 1)]:
        for qh in group:
            expected, _ = scaled_dot_product_attention(q[0, :, qh], k[0, :, kv_idx], v[0, :, kv_idx])
            assert np.allclose(out[0, :, qh], expected)


def test_multi_query_attention_is_gqa_with_one_kv_head():
    batch, seq, d_head = 1, 4, 4
    num_q_heads = 4
    q = np.random.default_rng(6).standard_normal((batch, seq, num_q_heads, d_head))
    k = np.random.default_rng(7).standard_normal((batch, seq, 1, d_head))
    v = np.random.default_rng(8).standard_normal((batch, seq, 1, d_head))

    out = grouped_query_attention(q, k, v, num_q_heads, 1)

    from attention_lib.sdpa import scaled_dot_product_attention
    for qh in range(num_q_heads):
        expected, _ = scaled_dot_product_attention(q[0, :, qh], k[0, :, 0], v[0, :, 0])
        assert np.allclose(out[0, :, qh], expected)


def test_gqa_rejects_indivisible_head_counts():
    q = np.random.randn(1, 3, 5, 4)
    k = np.random.randn(1, 3, 2, 4)
    v = np.random.randn(1, 3, 2, 4)
    with pytest.raises(AssertionError):
        grouped_query_attention(q, k, v, 5, 2)


def test_gqa_causal_masks_future():
    batch, seq, d_head = 1, 5, 4
    num_q_heads, num_kv_heads = 4, 2
    q = np.random.default_rng(9).standard_normal((batch, seq, num_q_heads, d_head))
    k = np.random.default_rng(10).standard_normal((batch, seq, num_kv_heads, d_head))
    v = np.random.default_rng(11).standard_normal((batch, seq, num_kv_heads, d_head))
    out_causal = grouped_query_attention(q, k, v, num_q_heads, num_kv_heads, causal=True)
    out_full = grouped_query_attention(q, k, v, num_q_heads, num_kv_heads, causal=False)
    assert not np.allclose(out_causal, out_full)
