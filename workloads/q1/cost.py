"""CPU time estimates for the compute stages, shown in notices and in preflight.

Rates were measured on one machine (8-core Apple laptop, numpy with Accelerate and torch 2.15,
2026-11-10, see harness/EMBEDDING_SITES.md section 5) on small samples. A server CPU with OpenBLAS
and more cores will usually scale better, so on such a machine these estimates are conservative.
Treat every estimate as an order of magnitude.
"""

import numpy as np

from workloads.q1.progress import format_duration

# numpy ground truth, one thread: float32 GFLOP/s of the whole candidate and recompute path, by dimension
# (small dimensions are limited by the top-k merge, not by the matmul)
GROUND_TRUTH_GFLOPS_ONE_THREAD = {100: 8.6, 128: 9.1, 384: 38.2, 768: 79.6}
GROUND_TRUTH_EFFICIENCY = 0.25    # speedup added per extra thread; measured 2.1x to 3.2x at 8 threads
# texts per second of one thread: ESCI product texts (MiniLM-L12 multilingual, about 60 tokens) and TriviaQA
# questions (DPR question encoder, about 12 words)
ENCODE_TEXTS_PER_SECOND_ONE_THREAD = {"esci": 70.0, "sphere": 80.0}
ENCODE_EFFICIENCY = 0.1           # measured 1.5x to 2x at 8 threads


def _speedup(threads: int, efficiency: float) -> float:
    return 1.0 + (max(threads, 1) - 1) * efficiency


def ground_truth_seconds(n_queries: int, n_base: int, dim: int, threads: int) -> float:
    dims = sorted(GROUND_TRUTH_GFLOPS_ONE_THREAD)
    rate = float(np.interp(dim, dims, [GROUND_TRUTH_GFLOPS_ONE_THREAD[d] for d in dims]))
    flops = 2.0 * n_queries * n_base * dim
    return flops / (rate * 1e9 * _speedup(threads, GROUND_TRUTH_EFFICIENCY))


def encode_seconds(dataset: str, n_texts: int, threads: int) -> float:
    rate = ENCODE_TEXTS_PER_SECOND_ONE_THREAD.get(dataset)
    if rate is None:
        return 0.0
    return n_texts / (rate * _speedup(threads, ENCODE_EFFICIENCY))


def describe(seconds: float) -> str:
    return f"about {format_duration(seconds)}"
