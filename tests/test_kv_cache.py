import numpy as np
import pytest

from attention_lib.kv_cache import KVCache, full_forward, incremental_forward_step


def make_weights(d_model, seed):
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(d_model)
    w_q = rng.standard_normal((d_model, d_model)) * scale
    w_k = rng.standard_normal((d_model, d_model)) * scale
    w_v = rng.standard_normal((d_model, d_model)) * scale
    w_o = rng.standard_normal((d_model, d_model)) * scale
    return w_q, w_k, w_v, w_o


@pytest.mark.parametrize("use_rope", [False, True])
def test_kv_cache_matches_full_forward_pass(use_rope):
    d_model, num_heads, seq_len, batch = 16, 4, 6, 2
    w_q, w_k, w_v, w_o = make_weights(d_model, seed=0)
    x = np.random.default_rng(1).standard_normal((batch, seq_len, d_model))

    full_out = full_forward(x, w_q, w_k, w_v, w_o, num_heads, use_rope=use_rope)

    cache = KVCache()
    incremental_outputs = []
    for t in range(seq_len):
        x_t = x[:, t:t + 1, :]
        out_t = incremental_forward_step(x_t, cache, w_q, w_k, w_v, w_o, num_heads, use_rope=use_rope)
        incremental_outputs.append(out_t)
    incremental_out = np.concatenate(incremental_outputs, axis=1)

    assert np.allclose(full_out, incremental_out, atol=1e-8)


def test_kv_cache_grows_correctly():
    cache = KVCache()
    assert cache.seq_len == 0
    k1 = np.random.randn(1, 2, 1, 4)
    v1 = np.random.randn(1, 2, 1, 4)
    cache.append(k1, v1)
    assert cache.seq_len == 1
    k2 = np.random.randn(1, 2, 1, 4)
    v2 = np.random.randn(1, 2, 1, 4)
    cache.append(k2, v2)
    assert cache.seq_len == 2
    assert np.allclose(cache.k[:, :, 0:1], k1)
    assert np.allclose(cache.k[:, :, 1:2], k2)


def test_kv_cache_single_token_matches_full_forward():
    d_model, num_heads = 8, 2
    w_q, w_k, w_v, w_o = make_weights(d_model, seed=5)
    x = np.random.default_rng(6).standard_normal((1, 1, d_model))
    full_out = full_forward(x, w_q, w_k, w_v, w_o, num_heads)
    cache = KVCache()
    inc_out = incremental_forward_step(x, cache, w_q, w_k, w_v, w_o, num_heads)
    assert np.allclose(full_out, inc_out, atol=1e-8)


def test_kv_cache_prefix_then_incremental_matches_full_forward():
    # Simulate: prefill first 3 tokens as a batch, then decode 2 more
    # one at a time, and check the final output matches a full forward
    # over all 5 tokens.
    d_model, num_heads = 8, 2
    w_q, w_k, w_v, w_o = make_weights(d_model, seed=11)
    x = np.random.default_rng(12).standard_normal((1, 5, d_model))

    full_out = full_forward(x, w_q, w_k, w_v, w_o, num_heads)

    cache = KVCache()
    prefill_out = incremental_forward_step(x[:, :3], cache, w_q, w_k, w_v, w_o, num_heads)
    assert cache.seq_len == 3
    assert np.allclose(prefill_out, full_out[:, :3], atol=1e-8)

    step_outs = [prefill_out]
    for t in range(3, 5):
        out_t = incremental_forward_step(x[:, t:t + 1], cache, w_q, w_k, w_v, w_o, num_heads)
        step_outs.append(out_t)
    combined = np.concatenate(step_outs, axis=1)
    assert np.allclose(combined, full_out, atol=1e-8)


def test_kv_cache_rope_absolute_positions_are_consistent():
    d_model, num_heads = 8, 2
    w_q, w_k, w_v, w_o = make_weights(d_model, seed=20)
    x = np.random.default_rng(21).standard_normal((1, 4, d_model))
    full_out = full_forward(x, w_q, w_k, w_v, w_o, num_heads, use_rope=True)

    cache = KVCache()
    outs = []
    for t in range(4):
        outs.append(incremental_forward_step(x[:, t:t + 1], cache, w_q, w_k, w_v, w_o, num_heads, use_rope=True))
    inc_out = np.concatenate(outs, axis=1)
    assert np.allclose(full_out, inc_out, atol=1e-8)


def test_kv_cache_different_batches_independent():
    d_model, num_heads = 8, 2
    w_q, w_k, w_v, w_o = make_weights(d_model, seed=30)
    x = np.random.default_rng(31).standard_normal((3, 4, d_model))
    full_out = full_forward(x, w_q, w_k, w_v, w_o, num_heads)

    cache = KVCache()
    outs = []
    for t in range(4):
        outs.append(incremental_forward_step(x[:, t:t + 1], cache, w_q, w_k, w_v, w_o, num_heads))
    inc_out = np.concatenate(outs, axis=1)
    assert np.allclose(full_out, inc_out, atol=1e-8)
