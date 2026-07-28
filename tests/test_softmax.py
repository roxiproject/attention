import numpy as np
import pytest

from attention_lib.functional import softmax


def test_softmax_sums_to_one_1d():
    x = np.array([1.0, 2.0, 3.0])
    y = softmax(x)
    assert np.isclose(np.sum(y), 1.0)


def test_softmax_sums_to_one_batched():
    x = np.random.randn(4, 5, 6)
    y = softmax(x, axis=-1)
    sums = np.sum(y, axis=-1)
    assert np.allclose(sums, 1.0)


def test_softmax_matches_naive_small_values():
    x = np.array([1.0, 2.0, 3.0])
    naive = np.exp(x) / np.sum(np.exp(x))
    assert np.allclose(softmax(x), naive)


def test_softmax_numerically_stable_large_values():
    x = np.array([1000.0, 1001.0, 1002.0])
    y = softmax(x)
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))
    assert np.isclose(np.sum(y), 1.0)


def test_softmax_numerically_stable_very_negative():
    x = np.array([-1000.0, -1001.0, -1002.0])
    y = softmax(x)
    assert not np.any(np.isnan(y))
    assert np.isclose(np.sum(y), 1.0)


def test_softmax_uniform_for_equal_inputs():
    x = np.array([5.0, 5.0, 5.0, 5.0])
    y = softmax(x)
    assert np.allclose(y, 0.25)


def test_softmax_axis_0():
    x = np.random.randn(3, 4)
    y = softmax(x, axis=0)
    sums = np.sum(y, axis=0)
    assert np.allclose(sums, 1.0)


def test_softmax_preserves_shape():
    x = np.random.randn(2, 3, 4, 5)
    y = softmax(x, axis=2)
    assert y.shape == x.shape


def test_softmax_all_values_nonnegative():
    x = np.random.randn(10, 10) * 100
    y = softmax(x)
    assert np.all(y >= 0)


def test_softmax_max_gets_highest_probability():
    x = np.array([0.1, 0.2, 10.0, 0.3])
    y = softmax(x)
    assert np.argmax(y) == 2
