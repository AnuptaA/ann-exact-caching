# Exact Vector Caching for Vector Databases

This repository supports research on cache management for vector databases. The central idea is to reuse exact results returned by underlying vector search systems whenever possible to reduce repeated work in top-K ANN query workloads without degrading the underlying recall of the vector search system. This work is index-agnostic and makes no assumptions about the underlying system, including what index it uses or if it even uses one at all.

The project started with proposing and theoretically proving correctness guarantees for two cache hit policies, and evolved into an empirical validation and analysis of them. It is now being extended toward cache population and eviction strategies to improve the performance of these hit policies on realistic workloads workloads. The goal is to introduce practical strategies providing correctness guarantees that appoximate caching methods do not, while also avoiding online-training and user-toned knobs.

## Research Artifacts

- Paper: [link coming soon!]
- Slides: [link coming soon!]

## Repository Layout

- `utils/base/`: Core distance metrics, cache entries, main-memory search, and hit-policy implementations.
- `datasets/`: Synthetic, SIFT, and ESCI data loading and benchmark preparation utilities.
- `simulations/`: Shared simulation engine plus workload-specific runners and analysis scripts.
- `tests/`: Unit, edge-case, and small end-to-end tests for the core algorithms and workloads.

## Current Research Direction

The existing code focuses on cache hit policies: given cached exact top-K results from prior queries, determine when a new query can be answered from cache while preserving recall of the underlying system. These policies are grounded in two guarantees we dub Circular Inclusion Guarantee (CIG) and Half-Gap Guarantee (HGG), and are evaluated across synthetic and realistic workloads.

The next phase is to extend the simulator and experiments to cache management policies, especially:

- How to populate the cache under realistic query streams.
- How to evict entries when cache capacity is limited.
- How hit-policy guarantees interact with population and eviction choices.
- How exact vector caching compares against existing approximate strategies, as well as how much performance benefit is provided in real systems.

## Current work: calibrated approximate caching

The current experiments relax exact hits to *approximate* hits with a calibrated guarantee: the
mean retention of hits (share of the backend's top-10 the cache returns) stays at or above a
target (0.9), with no per-workload tuning. It is compared against LRU, Aker (the authors' code) and
QVCache, by hit rate at the target (headline plot) and by end-to-end latency in front of the same
tuned DiskANN index (timing plot).

- How to reproduce both plots: `simulations/approximate_hits/README.md`
- Decisions, results and the full history: `docs/approximate_cache_hits_context.md`

## Getting Started

Exact-caching hit policies (CIG / HGG): see `utils/base/` and `tests/`
(`python -m pytest tests/`). Approximate caching: `simulations/approximate_hits/README.md`.
