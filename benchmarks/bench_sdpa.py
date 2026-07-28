"""Wall-clock comparison of a naive, explicit-loop scaled dot-product
attention against the batched/vectorized numpy implementation.

Run with:
    python benchmarks/bench_sdpa.py
"""
import time
import numpy as np

from attention_lib.sdpa import scaled_dot_product_attention


def naive_sdpa(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """O(seq^2 * d) attention computed with explicit Python-level loops
    over query and key positions (no numpy matmul for the score matrix).
    """
    seq_len_q, d_k = q.shape
    seq_len_k = k.shape[0]
    d_v = v.shape[1]
    out = np.zeros((seq_len_q, d_v))
    for i in range(seq_len_q):
        scores = np.empty(seq_len_k)
        for j in range(seq_len_k):
            scores[j] = np.dot(q[i], k[j]) / np.sqrt(d_k)
        m = np.max(scores)
        exp_scores = np.exp(scores - m)
        probs = exp_scores / np.sum(exp_scores)
        acc = np.zeros(d_v)
        for j in range(seq_len_k):
            acc += probs[j] * v[j]
        out[i] = acc
    return out


def vectorized_sdpa(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    out, _ = scaled_dot_product_attention(q, k, v)
    return out


def time_fn(fn, *args, repeats: int = 5) -> float:
    # one warmup run, then take the best of `repeats` timed runs
    fn(*args)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - start)
    return min(times)


def main():
    print(f"{'seq_len':>8} {'d_model':>8} {'naive (s)':>12} {'vectorized (s)':>16} {'speedup':>10}")
    for seq_len in (32, 64, 128, 256):
        for d_model in (64,):
            rng = np.random.default_rng(0)
            q = rng.standard_normal((seq_len, d_model))
            k = rng.standard_normal((seq_len, d_model))
            v = rng.standard_normal((seq_len, d_model))

            out_naive = naive_sdpa(q, k, v)
            out_vec = vectorized_sdpa(q, k, v)
            assert np.allclose(out_naive, out_vec, atol=1e-8), "implementations disagree"

            t_naive = time_fn(naive_sdpa, q, k, v, repeats=3)
            t_vec = time_fn(vectorized_sdpa, q, k, v, repeats=10)
            speedup = t_naive / t_vec if t_vec > 0 else float("inf")
            print(f"{seq_len:>8} {d_model:>8} {t_naive:>12.6f} {t_vec:>16.6f} {speedup:>9.1f}x")


if __name__ == "__main__":
    main()
