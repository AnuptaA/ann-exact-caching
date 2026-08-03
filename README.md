# Exact Vector Caching for Vector Databases

This repository supports research on cache management for vector databases. The central idea is to reuse exact results returned by underlying vector search systems whenever possible to reduce repeated work in top-K ANN query workloads without degrading the underlying recall of the vector search system. This work is index-agnostic and makes no assumptions about the underlying system, including what index it uses or if it even uses one at all.

The project started with proposing and theoretically proving correctness guarantees for two cache hit policies, and evolved into an empirical validation and analysis of them. It is now being extended toward cache population and eviction strategies to improve the performance of these hit policies on realistic workloads. The goal is to introduce practical strategies providing correctness guarantees that approximate caching methods do not, while also avoiding online-training and user-toned knobs.

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

## Getting Started
Instructions coming soon!
