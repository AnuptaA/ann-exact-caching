# Optimizing Cache Entry Selection (`selection_fast.py`)

## Summary

The original `select_candidates` in `selection.py` was slow because it used Python `set` objects for all its core operations. A new `select_candidates_fast` in `selection_fast.py` replaces those sets with numpy arrays, giving 7-18x speedup while producing identical results.

## The Problem

When you run a greedy selection strategy, the algorithm loops like this:

```
repeat until budget exhausted:
    for each remaining candidate:
        1. check: does this candidate fit in the budget?
        2. check: does this candidate cover any new queries?
    pick the best candidate
    add it to the selected set
```

Steps 1 and 2 are where all the time goes, because they're computed **for every candidate, every iteration**.

### Why Python sets are slow here

In the original code, `R_c` (the set of base vector indices inside a candidate's ball) is a Python `set` with ~661 elements on average. The budget check does:

```python
len(cached_vector_set | c.R_c) <= budget_B
```

This creates a **brand new temporary set** (the union) just to count its length. With 6000 candidates and 69 iterations, that's ~414,000 set unions, each allocating and populating a set of hundreds to thousands of integers. Python sets have high per-element overhead (each integer is a full Python object, ~28 bytes, stored in a hash table).

Similarly, the benefit check:

```python
len(c.covered_train_indices - covered_train_set)
```

creates a new set difference object for every candidate on every iteration.

## The Fix

### Idea 1: Use a numpy bool array for "what's cached"

Instead of a Python `set` of cached vector IDs, keep a flat boolean array:

```python
# Old: cached_vector_set = {3, 17, 42, ...}  (Python set)
# New: cached_bool = [False, False, False, True, ..., True, ...]  (numpy array, length 50000)
```

To check how many new vectors a candidate would add to the cache:

```python
# Old: len(cached_vector_set | c.R_c) - len(cached_vector_set)
#   -> creates a whole new set just to count it

# New: np.count_nonzero(~cached_bool[candidate_indices])
#   -> fancy-indexes into the bool array, counts False entries
#   -> no allocation, runs in C
```

### Idea 2: Use a numpy bool matrix for train/test coverage

Instead of each candidate having a Python `set` of covered query indices, build a matrix:

```python
# train_masks: shape (6000 candidates, 1024 train queries), dtype=bool
# Each row has True where that candidate covers a query
```

To compute "how many new queries does each alive candidate cover?":

```python
# Old: loop over candidates, compute len(c.covered_train_indices - covered_train_set)
# New (one line, ALL candidates at once):
newly = np.count_nonzero(train_masks[alive_idx] & ~covered_train, axis=1)
```

This single numpy call replaces thousands of Python set-difference operations.

### Idea 3: Batch the marginal cost computation

For budget checks, all alive candidates' `R_c` index arrays are concatenated into one flat array. A single fancy-index lookup (`cached_bool[flat_array]`) fetches all the answers, and `np.add.reduceat` splits the result back into per-candidate counts. This avoids Python loop overhead for the budget check.

## What Stays the Same

- The original `selection.py` is untouched and still works.
- `select_candidates_fast` has the exact same function signature and returns identical `(selected, trajectory)` tuples.
- All four strategies (`random`, `greedy_naive`, `greedy_cost_static`, `greedy_cost`) are implemented.

## Timing Results

Measured on synthetic data designed to match real SIFT benchmark characteristics:
- 6000 candidates (1024 queries x ~6 alpha values producing valid candidates)
- Median R_c size: 661 base vectors per candidate
- Median coverage: 3 train queries per candidate
- Budget: 10,000 base vectors
- n_base: 50,000 | n_train: 1,024 | n_test: 512

**Caveat: this is on synthetic (fake) data with realistic size distributions, not actual SIFT/ESCI embeddings. Real runs may differ somewhat, but the speedup ratios should hold.**

| Strategy | Original | Fast | Speedup |
|----------|----------|------|---------|
| `greedy_naive` | 8.7s | 1.3s | 6.8x |
| `greedy_cost_static` | 37.6s | 2.8s | 13.2x |
| `greedy_cost` | 51.5s | 2.8s | 18.3x |

The costlier strategies benefit more because they run more iterations (selecting more entries before the budget fills), so the per-iteration savings compound.

## How to Use

```python
from utils.base.selection_fast import select_candidates_fast

# Same interface as select_candidates
selected, trajectory = select_candidates_fast(
    candidates, budget_B, strategy,
    n_train_queries=n_train,
    n_test_queries=n_test,
    seed=42,
)
```

## What Could Still Be Improved

The remaining ~2-3 seconds is mostly memory-access cost: each iteration looks up ~3 million random positions in numpy arrays. Going faster would require:

1. **Numba JIT compilation** — would eliminate remaining Python overhead and allow fused loops
2. **Incremental marginal updates** — after selecting a candidate, only re-check candidates whose R_c overlaps with the one just selected (algorithmic optimization, more complex to implement)

These are diminishing returns and would add complexity, so the current version is a good stopping point unless runs need to be sub-second.
