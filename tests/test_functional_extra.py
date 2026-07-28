import numpy as np

from attention_lib.functional import softmax
from attention_lib.sdpa import scaled_dot_product_attention


def test_softmax_gradient_free_no_side_effects_on_input():
    x = np.array([1.0, 2.0, 3.0])
    x_copy = x.copy()
    _ = softmax(x)
    assert np.array_equal(x, x_copy)


def test_softmax_output_dtype_float():
    x = np.array([1, 2, 3])
    y = softmax(x)
    assert np.issubdtype(y.dtype, np.floating)


def test_softmax_monotonic_with_input():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = softmax(x)
    assert np.all(np.diff(y) > 0)


def test_sdpa_does_not_mutate_inputs():
    q = np.random.randn(3, 4)
    k = np.random.randn(5, 4)
    v = np.random.randn(5, 6)
    q0, k0, v0 = q.copy(), k.copy(), v.copy()
    scaled_dot_product_attention(q, k, v)
    assert np.array_equal(q, q0)
    assert np.array_equal(k, k0)
    assert np.array_equal(v, v0)


def test_sdpa_scaling_reduces_variance_of_scores():
    # With scaling, dot products don't blow up with dimension, keeping
    # softmax from becoming a hard argmax for random inputs.
    rng = np.random.default_rng(0)
    d_k = 512
    q = rng.standard_normal((1, d_k))
    k = rng.standard_normal((20, d_k))
    v = rng.standard_normal((20, 3))
    _, weights = scaled_dot_product_attention(q, k, v)
    # not fully collapsed onto one key
    assert np.max(weights) < 0.999


def test_sdpa_linearity_in_v():
    rng = np.random.default_rng(1)
    q = rng.standard_normal((2, 4))
    k = rng.standard_normal((3, 4))
    v1 = rng.standard_normal((3, 5))
    v2 = rng.standard_normal((3, 5))
    out1, w1 = scaled_dot_product_attention(q, k, v1)
    out2, w2 = scaled_dot_product_attention(q, k, v2)
    out_sum, w_sum = scaled_dot_product_attention(q, k, v1 + v2)
    assert np.allclose(w1, w2)  # weights depend only on q, k
    assert np.allclose(w1, w_sum)
    assert np.allclose(out1 + out2, out_sum, atol=1e-8)
