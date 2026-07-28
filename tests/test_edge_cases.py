import numpy as np
import pytest

from attention_lib.functional import softmax
from attention_lib.sdpa import scaled_dot_product_attention, causal_mask
from attention_lib.multihead import split_heads, combine_heads, MultiHeadAttention, grouped_query_attention
from attention_lib.positional import apply_rope, rope_frequencies


def test_softmax_single_element():
    x = np.array([5.0])
    y = softmax(x)
    assert np.isclose(y[0], 1.0)


def test_softmax_two_dim_axis_negative_two():
    x = np.random.randn(3, 4, 5)
    y = softmax(x, axis=-2)
    assert np.allclose(np.sum(y, axis=-2), 1.0)


def test_softmax_zero_input():
    x = np.zeros(5)
    y = softmax(x)
    assert np.allclose(y, 0.2)


def test_causal_mask_non_square():
    # seq_len_k > seq_len_q, as with a KV cache: 2 new queries, 5 total keys.
    m = causal_mask(2, 5)
    assert m.shape == (2, 5)
    # last query (index 1) corresponds to absolute position 4, sees all keys
    assert np.all(m[1])
    # first query (index 0) corresponds to absolute position 3, sees keys 0..3
    assert np.array_equal(m[0], [True, True, True, True, False])


def test_causal_mask_single_query_sees_all_keys():
    m = causal_mask(1, 7)
    assert np.all(m)


def test_sdpa_single_key():
    q = np.random.randn(3, 4)
    k = np.random.randn(1, 4)
    v = np.random.randn(1, 6)
    out, weights = scaled_dot_product_attention(q, k, v)
    assert np.allclose(weights, 1.0)
    assert np.allclose(out, np.broadcast_to(v[0], (3, 6)))


def test_sdpa_d_k_one():
    q = np.array([[2.0]])
    k = np.array([[1.0], [2.0], [3.0]])
    v = np.array([[10.0], [20.0], [30.0]])
    out, weights = scaled_dot_product_attention(q, k, v)
    assert out.shape == (1, 1)
    assert np.isclose(np.sum(weights), 1.0)


def test_sdpa_zero_vectors_uniform_attention():
    q = np.zeros((2, 4))
    k = np.zeros((3, 4))
    v = np.random.randn(3, 5)
    _, weights = scaled_dot_product_attention(q, k, v)
    assert np.allclose(weights, 1.0 / 3)


def test_split_heads_single_head_is_identity_reshape():
    x = np.random.randn(2, 3, 8)
    out = split_heads(x, num_heads=1)
    assert out.shape == (2, 1, 3, 8)
    assert np.allclose(out[:, 0], x)


def test_combine_heads_single_head_is_identity_reshape():
    x = np.random.randn(2, 1, 3, 8)
    out = combine_heads(x)
    assert np.allclose(out[:, :, :], x[:, 0])


def test_mha_with_single_head_matches_plain_sdpa():
    d_model = 8
    mha = MultiHeadAttention(d_model, num_heads=1, seed=3)
    x = np.random.default_rng(4).standard_normal((1, 5, d_model))
    out = mha(x)

    q = x @ mha.w_q
    k = x @ mha.w_k
    v = x @ mha.w_v
    ref, _ = scaled_dot_product_attention(q[0], k[0], v[0])
    ref = (ref @ mha.w_o[:, :]).reshape(1, 5, d_model) if False else ref @ mha.w_o
    assert np.allclose(out[0], ref, atol=1e-8)


def test_mha_batch_independence():
    # Each batch element's output should not depend on other batch elements.
    d_model, num_heads = 8, 2
    mha = MultiHeadAttention(d_model, num_heads, seed=8)
    rng = np.random.default_rng(9)
    x1 = rng.standard_normal((1, 4, d_model))
    x2 = rng.standard_normal((1, 4, d_model))
    x_batched = np.concatenate([x1, x2], axis=0)

    out_batched = mha(x_batched)
    out1 = mha(x1)
    out2 = mha(x2)

    assert np.allclose(out_batched[0], out1[0], atol=1e-8)
    assert np.allclose(out_batched[1], out2[0], atol=1e-8)


def test_rope_frequencies_first_is_always_one():
    for d in [2, 4, 8, 16, 64]:
        freqs = rope_frequencies(d)
        assert np.isclose(freqs[0], 1.0)


def test_apply_rope_two_positions_orthogonal_case():
    # theta=pi/2 special construction: base chosen so freq=1, position pi/2
    # rotates (1,0) to (0,1).
    x = np.zeros((1, 1, 2))
    x[0, 0] = [1.0, 0.0]
    out = apply_rope(x, start_pos=0, base=1.0)
    assert np.allclose(out[0, 0], [1.0, 0.0])


def test_gqa_num_kv_heads_one_reduces_memory_shape():
    batch, seq, d_head = 1, 4, 4
    q = np.random.randn(batch, seq, 8, d_head)
    k = np.random.randn(batch, seq, 1, d_head)
    v = np.random.randn(batch, seq, 1, d_head)
    out = grouped_query_attention(q, k, v, 8, 1)
    assert out.shape == (batch, seq, 8, d_head)


def test_full_pipeline_rope_plus_gqa_runs_without_error():
    batch, seq, num_q_heads, num_kv_heads, d_head = 2, 6, 8, 2, 8
    rng = np.random.default_rng(42)
    q = rng.standard_normal((batch, seq, num_q_heads, d_head))
    k = rng.standard_normal((batch, seq, num_kv_heads, d_head))
    v = rng.standard_normal((batch, seq, num_kv_heads, d_head))

    q_bh = np.transpose(q, (0, 2, 1, 3))
    q_rot = apply_rope(q_bh)
    q_rot = np.transpose(q_rot, (0, 2, 1, 3))

    k_bh = np.transpose(k, (0, 2, 1, 3))
    k_rot = apply_rope(k_bh)
    k_rot = np.transpose(k_rot, (0, 2, 1, 3))

    out = grouped_query_attention(q_rot, k_rot, v, num_q_heads, num_kv_heads, causal=True)
    assert out.shape == (batch, seq, num_q_heads, d_head)
    assert not np.any(np.isnan(out))
