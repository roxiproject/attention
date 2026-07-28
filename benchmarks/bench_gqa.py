"""Wall-clock comparison of standard multi-head attention (one KV head per
query head) against grouped-query attention (few shared KV heads), showing
the compute/memory savings from sharing KV projections.

Run with:
    python benchmarks/bench_gqa.py
"""
import time
import numpy as np

from attention_lib.multihead import grouped_query_attention


def time_fn(fn, *args, repeats: int = 10) -> float:
    fn(*args)  # warmup
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - start)
    return min(times)


def main():
    batch, seq, d_head = 4, 512, 64
    num_query_heads = 16
    rng = np.random.default_rng(0)
    q = rng.standard_normal((batch, seq, num_query_heads, d_head))

    print(f"{'num_kv_heads':>14} {'kv memory (elems)':>20} {'time (s)':>12}")
    for num_kv_heads in (16, 8, 4, 2, 1):
        k = rng.standard_normal((batch, seq, num_kv_heads, d_head))
        v = rng.standard_normal((batch, seq, num_kv_heads, d_head))
        t = time_fn(grouped_query_attention, q, k, v, num_query_heads, num_kv_heads)
        kv_elems = k.size + v.size
        print(f"{num_kv_heads:>14} {kv_elems:>20} {t:>12.6f}")


if __name__ == "__main__":
    main()
