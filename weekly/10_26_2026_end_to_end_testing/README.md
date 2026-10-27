# End-to-end cache pipeline tests (branch 10__26_2026_end_to_end_testing)

Cache population plus cache hit policy, with an oracle radius arm and an LSH radius arm. K = 10 (this folder only; `weekly/9_22_2026_hit_policy_testing/common/config.py` has K = 100 and is not used for K).

## Layout

```
end_to_end/   code (prefix ee_) and tests (test_ee_*.py)
results/      run tables, ground truth cache (results/ground_truth), logs (results/logs)
plots/  pdf/  figures and reports
```
`results/`, `plots/` and `pdf/` are covered by the repo `.gitignore` patterns `weekly/*/results/`, `plots/`, `pdf/`.
The folder name follows `M_DD_YYYY_topic` like `9_22_2026_hit_policy_testing`. The branch name has a double underscore (`10__26`); it is kept as is.

## Reused without edits

From `weekly/9_22_2026_hit_policy_testing/` (added to `sys.path` by `ee_config.py`): `common/` (`data_prep`, `ground_truth`, `logging_utils`, `perturbation`, `plot_style`) and `radius_eval_tables/` (`rt_chunks`, `rt_engine`).
Anything that has to change is copied into `end_to_end/`. The old threshold_baseline population policy was copied and adapted from `population.py` of the deleted 09_22 hit-policy pipeline (kept only in a local tarball).

## Design notes

- Workloads: lambda in {0, 0.5, 1.0} is the local fraction. The anchors are nested across lambda for a given seed (the lambda = 0.5 anchors are the first 25 of the lambda = 1.0 anchors), but the random
  component is not nested: lower lambda means more random queries, drawn separately per lambda. Comparisons across lambda are therefore not between identical random queries.
- At lambda = 1.0 there are 50 anchors with 50 train and 50 test copies each (2,500 local queries, no random ones).
- Recall is computed on integer units: with K = 10 every estimated recall is a multiple of 0.1, and the hit policies compare in units of 0.05 recall, so no float tolerance is needed.
- Seeds 0 to 4 are used for reported runs, seeds 100 to 102 only for the LSH mini-benchmark.

## Run (from the thesis root)

```
python3 -B weekly/10_26_2026_end_to_end_testing/end_to_end/test_ee_<module>.py
```
