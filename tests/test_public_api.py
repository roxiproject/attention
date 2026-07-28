import numpy as np

import attention_lib as al


def test_package_exposes_softmax():
    x = np.array([1.0, 2.0, 3.0])
    y = al.softmax(x)
    assert np.isclose(np.sum(y), 1.0)


def test_package_exposes_sdpa():
    q = np.random.randn(3, 4)
    k = np.random.randn(5, 4)
    v = np.random.randn(5, 6)
    out, weights = al.scaled_dot_product_attention(q, k, v)
    assert out.shape == (3, 6)
    assert np.allclose(np.sum(weights, axis=-1), 1.0)


def test_package_exposes_multihead_attention():
    mha = al.MultiHeadAttention(8, 2, seed=0)
    x = np.random.randn(1, 4, 8)
    out = mha(x)
    assert out.shape == (1, 4, 8)


def test_package_exposes_grouped_query_attention():
    q = np.random.randn(1, 3, 4, 4)
    k = np.random.randn(1, 3, 2, 4)
    v = np.random.randn(1, 3, 2, 4)
    out = al.grouped_query_attention(q, k, v, 4, 2)
    assert out.shape == (1, 3, 4, 4)


def test_package_exposes_rope():
    x = np.random.randn(1, 3, 8)
    out = al.apply_rope(x)
    assert out.shape == x.shape


def test_package_exposes_kv_cache():
    cache = al.KVCache()
    assert cache.seq_len == 0
    k = np.random.randn(1, 2, 1, 4)
    v = np.random.randn(1, 2, 1, 4)
    cache.append(k, v)
    assert cache.seq_len == 1


def test_package_version_string():
    assert isinstance(al.__version__, str)
    assert al.__version__.count(".") == 2
