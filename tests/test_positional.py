import numpy as np
import pytest

from attention_lib.positional import (
    rope_frequencies,
    rope_angles,
    apply_rope,
    relative_position_bias,
)


def test_rope_frequencies_length():
    freqs = rope_frequencies(8)
    assert freqs.shape == (4,)


def test_rope_frequencies_known_values_base_10000_d4():
    # d_head = 4 -> i in {0, 2} -> theta = 10000^(-i/4) -> [1.0, 10000^-0.5]
    freqs = rope_frequencies(4, base=10000.0)
    expected = np.array([1.0, 10000.0 ** (-0.5)])
    assert np.allclose(freqs, expected)


def test_rope_frequencies_decreasing():
    freqs = rope_frequencies(16)
    assert np.all(np.diff(freqs) < 0)


def test_rope_frequencies_rejects_odd_dim():
    with pytest.raises(AssertionError):
        rope_frequencies(5)


def test_rope_angles_position_zero_is_zero():
    angles = rope_angles(5, 8)
    assert np.allclose(angles[0], 0.0)


def test_rope_angles_shape():
    angles = rope_angles(6, 8)
    assert angles.shape == (6, 4)


def test_apply_rope_manual_2d_case():
    # d_head = 2, single frequency theta_0 = 1.0 (base^0 = 1).
    # Position 0: angle 0 -> identity rotation.
    # Position 1: angle 1 rad -> standard 2D rotation matrix.
    x = np.array([[[1.0, 0.0], [1.0, 0.0]]])  # shape (1, 2, 2): batch=1, seq=2, d_head=2
    out = apply_rope(x, base=1.0)  # base=1 -> theta = 1**0 = 1 for all, freq=1
    # position 0: angle=0 -> (1,0) unchanged
    assert np.allclose(out[0, 0], [1.0, 0.0], atol=1e-10)
    # position 1: angle=1 rad, rotate (1,0) by 1 rad -> (cos1, sin1)
    expected = np.array([np.cos(1.0), np.sin(1.0)])
    assert np.allclose(out[0, 1], expected, atol=1e-10)


def test_apply_rope_manual_known_vector():
    # Independently hand-computed: d_head=2, base=10000 (irrelevant, single
    # freq = 1 since i=0), position=2, vector (0, 1).
    # angle = 2 * 1.0 = 2 rad. Rotation: x1' = x1*cos - x2*sin, x2' = x1*sin + x2*cos
    # = (0*cos2 - 1*sin2, 0*sin2 + 1*cos2) = (-sin2, cos2)
    x = np.zeros((1, 3, 2))
    x[0, 2] = [0.0, 1.0]
    out = apply_rope(x)
    expected = np.array([-np.sin(2.0), np.cos(2.0)])
    assert np.allclose(out[0, 2], expected, atol=1e-10)


def test_apply_rope_preserves_vector_norm():
    # Rotation preserves the L2 norm of each 2D pair.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 5, 8))
    out = apply_rope(x)
    for b in range(2):
        for s in range(5):
            pairs_in = x[b, s].reshape(-1, 2)
            pairs_out = out[b, s].reshape(-1, 2)
            norms_in = np.linalg.norm(pairs_in, axis=-1)
            norms_out = np.linalg.norm(pairs_out, axis=-1)
            assert np.allclose(norms_in, norms_out, atol=1e-8)


def test_apply_rope_position_zero_is_identity():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 1, 8))
    out = apply_rope(x)
    assert np.allclose(out, x)


def test_apply_rope_shape_preserved():
    x = np.random.randn(2, 3, 4, 16)
    out = apply_rope(x)
    assert out.shape == x.shape


def test_apply_rope_start_pos_offsets_rotation():
    x = np.random.default_rng(2).standard_normal((1, 1, 8))
    out_pos5 = apply_rope(x, start_pos=5)
    full = np.tile(x, (1, 6, 1))
    out_full = apply_rope(full)
    assert np.allclose(out_pos5[0, 0], out_full[0, 5], atol=1e-8)


def test_apply_rope_relative_position_property():
    # Key correctness property of RoPE: dot(rope(q, m), rope(k, n)) depends
    # only on (m - n), not on m, n individually.
    rng = np.random.default_rng(3)
    q = rng.standard_normal(8)
    k = rng.standard_normal(8)

    def dot_at(m, n):
        q_rot = apply_rope(q.reshape(1, 1, 8), start_pos=m)[0, 0]
        k_rot = apply_rope(k.reshape(1, 1, 8), start_pos=n)[0, 0]
        return np.dot(q_rot, k_rot)

    d1 = dot_at(3, 1)   # relative distance 2
    d2 = dot_at(10, 8)  # relative distance 2
    d3 = dot_at(5, 5)   # relative distance 0
    assert np.isclose(d1, d2, atol=1e-8)
    assert not np.isclose(d1, d3, atol=1e-3)


def test_relative_position_bias_shape():
    bias = relative_position_bias(5, 5)
    assert bias.shape == (5, 5)


def test_relative_position_bias_diagonal_is_small_bucket():
    bias = relative_position_bias(6, 6)
    # diagonal (relative_position == 0) should map to bucket 0
    assert np.all(np.diag(bias) == 0)


def test_relative_position_bias_symmetric_sign_split():
    bias = relative_position_bias(8, 8, num_buckets=32)
    # positions after the query (k_idx > q_idx) use the "positive" half of buckets
    upper = bias[0, 1:]
    lower = bias[7, :7]
    assert np.all(upper >= 16)
    assert np.all(lower < 16)
