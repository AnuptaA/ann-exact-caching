#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Orchestrator for the multi-table radius experiments.

Run from the thesis root:
    python3 weekly/9_22_2026_hit_policy_testing/radius_eval_tables/rt_run.py \
        --datasets both --stages 0 [--dry-run]
Prefer rt_supervise.py for long runs (caffeinate plus automatic resume)."""

import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import rt_chunks
import rt_config
import rt_data
import rt_engine
import rt_exact
from logging_utils import setup_logging

DRY_N = 100
DRY_SEEDS = [0, 1]

#-------------------------------------------------------------------------------

def constant_for(theta_db: np.ndarray, seed: int, S: int) -> float:
    return float(np.median(theta_db[rt_data.sample_permutation(len(theta_db), seed)[:S]]))

def query_digest(qs: dict) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(qs["train"]).tobytes())
    h.update(np.ascontiguousarray(qs["test"]).tobytes())
    return h.hexdigest()

#-------------------------------------------------------------------------------

def _exact_agg_row(theta: np.ndarray, r_hat: np.ndarray, config: str, S: int, pop: str, seed: int, nu: int) -> dict:
    row = dict(config=config, L=0, m=0, S=S, pop=pop, nu=nu, seed=seed, **rt_engine._stat_sums(theta, r_hat))
    row.update(distinct_scope="none", n_distinct=np.nan, sum_distinct_total=np.nan, sum_distinct_train=np.nan,
               sum_distinct_sample=np.nan, sum_train_share=np.nan)
    return row

def consolidate(store: rt_chunks.ChunkStore, out_prefix: Path, plan: list, cov: pd.DataFrame, theta_db: np.ndarray,
                seeds: list, has_dump: bool, fingerprint: str, logger, summary: dict, s_grid: list,
                ref_S: int) -> None:
    rows = pd.concat([rt_chunks.read_table(store.name(s, c, "rows.csv.gz")) for s, c in plan], ignore_index=True)
    test = cov[cov["role"] == "test"].reset_index(drop=True)
    theta_test = test["theta"].to_numpy()
    theta_by_id = dict(zip(test["query_id"], theta_test))
    nan_cols = {c: np.nan for c in rows.columns if c not in ("query_id", "seed", "config", "nu", "r_hat")}
    exact = pd.DataFrame(dict(query_id=test["query_id"], seed=-1, config="EXACT", nu=0, r_hat=theta_test, **nan_cols))
    max_err = float(np.abs(exact["r_hat"].to_numpy() - theta_test).max())
    assert max_err == 0.0, "EXACT error must be exactly 0"
    consts = []
    for seed in seeds:
        cv = constant_for(theta_db, seed, ref_S)
        consts.append(pd.DataFrame(dict(query_id=test["query_id"], seed=seed, config="CONSTANT", nu=0,
                                        r_hat=cv, **nan_cols)))
    rt_chunks.write_table_atomic(pd.concat([rows, exact] + consts, ignore_index=True),
                                 Path(f"{out_prefix}_per_query_ref.csv.gz"))
    agg = pd.concat([rt_chunks.read_table(store.name(s, c, "agg.csv.gz")) for s, c in plan], ignore_index=True)
    keys = ["config", "L", "m", "S", "pop", "nu", "seed", "distinct_scope"]
    sums = [c for c in agg.columns if c not in keys]
    agg = agg.groupby(keys, sort=False)[sums].sum(min_count=1).reset_index()
    extra = [_exact_agg_row(theta_test, theta_test, "EXACT", -1, "none", -1, 0)]
    for seed in seeds:
        for S in s_grid:
            cv = constant_for(theta_db, seed, S)
            extra.append(_exact_agg_row(theta_test, np.full(len(theta_test), cv), "CONSTANT", S, "sample_only", seed, 0))
    agg = pd.concat([agg, pd.DataFrame(extra)], ignore_index=True)
    rt_chunks.write_table_atomic(agg, Path(f"{out_prefix}_aggregates.csv.gz"))
    if has_dump:
        dump = pd.concat([rt_chunks.read_table(store.name(0, c, "dump.csv.gz")) for s, c in plan
                          if s == 0 and store.name(0, c, "dump.csv.gz").exists()], ignore_index=True)
        header = [f"neighbor dump: seed 0, S = {ref_S}, sample+train, one row per retrieved neighbor per table",
                  f"rank_in_table is 1-based by (hamming, labeled_id); the {rt_config.DUMP_NU_MAX} smallest per table",
                  f"ties at the cutoff are capped at {rt_config.DUMP_TIE_CAP} rows per table, keeping the smallest labeled ids",
                  f"labeled_id: database index for sample points, {summary['N']} + train index for train points",
                  "from_train: 1 for train queries, 0 for sampled database vectors"]
        rt_chunks.write_table_atomic(dump, Path(f"{out_prefix}_neighbor_dump.csv.gz"), header_comments=header)
    summary.update(exact_max_abs_error=max_err, per_query_rows=int(len(rows)), aggregate_rows=int(len(agg)),
                   finished=datetime.now().isoformat())
    Path(f"{out_prefix}_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    rt_chunks.marker_path(out_prefix).write_text(json.dumps(dict(fingerprint=fingerprint,
                                                                 written=datetime.now().isoformat())))
    logger.info(f"consolidated {len(plan)} chunk files -> {out_prefix.name}_*; removing {store.dir.name}")
    store.remove()

#-------------------------------------------------------------------------------

def run_stage(dataset: str, stage: int, seeds: list, n_train: int, n_test: int, out_dir: Path,
              chunk_log: rt_chunks.ChunkLog, logger, rss_limit_mb: float, min_free_disk_gb: float,
              chunk_size: int = rt_config.CHUNK_SIZE) -> dict:
    t_stage = time.time()
    label = f"{dataset}/stage{stage}"
    free = rt_chunks.free_disk_gb(out_dir)
    if free < min_free_disk_gb:
        raise rt_chunks.RunAbort(f"free disk {free:.2f} GiB below {min_free_disk_gb} GiB at {label} start")
    timings = {}
    t0 = time.time()
    qs = rt_data.build_stage(dataset, stage, logger, n_train, n_test)
    database = qs["database"]
    N = len(database)
    timings["build_queries"] = time.time() - t0
    t0 = time.time()
    theta_db, cache_hit = rt_data.build_or_load_theta(qs["db_key"], database, logger)
    timings["theta_cache"] = time.time() - t0
    fp = rt_chunks.run_fingerprint(dataset, stage, seeds, n_train, n_test, rt_config.CONFIGS, rt_config.NU_GRID,
                                   rt_config.S_GRID, chunk_size, N, query_digest(qs))
    out_prefix = out_dir / f"{dataset}_stage{stage}"
    if rt_chunks.stage_complete(out_prefix, fp):
        logger.info(f"{label}: already complete (fingerprint {fp}), skipping")
        return dict(skipped=True)
    store = rt_chunks.ChunkStore(out_dir, dataset, stage, fp)

    t0 = time.time()
    both = np.vstack([qs["train"], qs["test"]])
    cov = rt_exact.query_covariates(both, database)
    meta = pd.concat([qs["meta_train"], qs["meta_test"]], ignore_index=True)
    cov = pd.concat([meta, cov], axis=1)
    assert cov["theta"].gt(0).all() and cov["nn1"].gt(rt_config.DUP_TOL).all()
    rt_chunks.write_table_atomic(cov, Path(f"{out_prefix}_covariates.csv.gz"))
    timings["exact_pass"] = time.time() - t0
    theta_train = cov["theta"].to_numpy()[:n_train]
    theta_test = cov["theta"].to_numpy()[n_train:]
    logger.info(f"{label}: exact pass done in {timings['exact_pass']:.0f}s, test theta mean {theta_test.mean():.4f} "
                f"sd {theta_test.std():.4f} (relative sd {100 * theta_test.std() / theta_test.mean():.2f}%)")

    has_dump = stage in rt_config.DUMP_STAGES
    dump_ids = np.sort(np.random.default_rng(rt_config.DATA_SEED + 7).choice(n_test, min(rt_config.DUMP_N_QUERIES, n_test),
                                                                          replace=False))
    n_chunks = int(np.ceil(n_test / chunk_size))
    n_cfg, n_nu = len(rt_config.CONFIGS), len(rt_config.NU_GRID)
    n_pops = len(rt_config.S_GRID) * len(rt_config.POP_KINDS)
    states, plan, stats = {}, [], dict(computed=0, skipped=0, peak_rss_mb=0.0, phases={})
    for ci in range(n_chunks):
        s, e = ci * chunk_size, min(n_test, (ci + 1) * chunk_size)
        local = np.flatnonzero((dump_ids >= s) & (dump_ids < e)) if has_dump else None
        local_rows = dump_ids[local] - s if has_dump else None
        for seed in seeds:
            plan.append((seed, ci))
            expect_dump = has_dump and seed == 0 and len(local_rows) > 0
            if store.is_valid(seed, ci, (e - s) * n_cfg * n_nu, n_cfg * n_pops * n_nu, expect_dump):
                stats["skipped"] += 1
                logger.info(chunk_log.write(dataset, stage, seed, ci, "SKIP", 0.0, rt_chunks.current_rss_mb()))
                continue
            if seed not in states:
                tb = time.time()
                states[seed] = rt_engine.build_seed_state(seed, database, theta_db, qs["train"], theta_train)
                timings["seed_state"] = timings.get("seed_state", 0.0) + time.time() - tb
            t0 = time.time()
            opts = rt_engine.ChunkOptions(dump_local=local_rows)
            out = rt_engine.process_chunk(states[seed], qs["test"][s:e], theta_test[s:e], np.arange(s, e), opts)
            reason = rt_chunks.integrity_reason(out["rows"], out["agg"])
            if reason:
                raise rt_chunks.RunAbort(f"{label} seed {seed} chunk {ci}: {reason} (chunk NOT written)")
            store.write_chunk(seed, ci, out["rows"], out["agg"], out["dump"])
            elapsed = time.time() - t0
            rss = rt_chunks.current_rss_mb()
            stats["computed"] += 1
            stats["peak_rss_mb"] = max(stats["peak_rss_mb"], rss)
            for k, v in out["timings"].items():
                stats["phases"][k] = stats["phases"].get(k, 0.0) + v
            logger.info(chunk_log.write(dataset, stage, seed, ci, "DONE", elapsed, rss, out["timings"]))
            del out
            gc.collect()
            if rss > rss_limit_mb:
                raise rt_chunks.RunAbort(f"RSS {rss:.0f} MB exceeds {rss_limit_mb:.0f} MB after {label} seed {seed} "
                                         f"chunk {ci} (chunk was written and flushed)")
            free = rt_chunks.free_disk_gb(out_dir)
            if free < min_free_disk_gb:
                raise rt_chunks.RunAbort(f"free disk {free:.2f} GiB below {min_free_disk_gb} GiB after {label} seed "
                                         f"{seed} chunk {ci} (chunk was written and flushed)")
    summary = dict(dataset=dataset, stage=stage, fingerprint=fp, N=N, n_train=n_train, n_test=n_test, seeds=seeds,
                   theta_cache_hit=bool(cache_hit), timings_seconds=timings, chunk_phase_seconds=stats["phases"],
                   chunks_computed=stats["computed"], chunks_skipped=stats["skipped"],
                   peak_rss_mb=stats["peak_rss_mb"], test_theta_mean=float(theta_test.mean()),
                   test_theta_relative_sd=float(theta_test.std() / theta_test.mean()))
    t0 = time.time()
    consolidate(store, out_prefix, plan, cov, theta_db, seeds, has_dump, fp, logger, summary, rt_config.S_GRID,
                rt_config.REF_S)
    summary["timings_seconds"]["consolidate"] = time.time() - t0
    summary["wall_seconds"] = time.time() - t_stage
    Path(f"{out_prefix}_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    logger.info(f"{label}: COMPLETE in {summary['wall_seconds']:.0f}s (peak RSS {stats['peak_rss_mb']:.0f} MB)")
    return summary

#-------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="both", choices=["sift", "esci", "both"])
    ap.add_argument("--stages", default="0,1,2,3,4")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rss-limit-mb", type=float, default=rt_config.RSS_LIMIT_MB)
    ap.add_argument("--min-free-disk-gb", type=float, default=rt_config.MIN_FREE_DISK_GB)
    ap.add_argument("--query-chunk-size", type=int, default=rt_config.CHUNK_SIZE)
    args = ap.parse_args()
    datasets = rt_config.DATASETS if args.datasets == "both" else [args.datasets]
    stages = [int(x) for x in args.stages.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else (DRY_SEEDS if args.dry_run else rt_config.SEEDS)
    n_train = DRY_N if args.dry_run else rt_config.N_TRAIN
    n_test = DRY_N if args.dry_run else rt_config.N_TEST
    out_dir = rt_config.DRY_RUN_DIR if args.dry_run else rt_config.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("rt_run_dry_run" if args.dry_run else "rt_run", rt_config.LOGS_DIR)
    chunk_log = rt_chunks.ChunkLog(rt_config.LOGS_DIR / ("chunk_log_dry_run.csv" if args.dry_run else "chunk_log.csv"))
    logger.info(f"RT_RUN STARTING dry_run={args.dry_run} datasets={datasets} stages={stages} seeds={seeds} "
                f"n_train={n_train} n_test={n_test} chunk={args.query_chunk_size}")
    summaries = []
    try:
        for stage in stages:
            for dataset in datasets:
                summaries.append(run_stage(dataset, stage, seeds, n_train, n_test, out_dir, chunk_log, logger,
                                           args.rss_limit_mb, args.min_free_disk_gb, args.query_chunk_size))
    except rt_chunks.RunAbort as exc:
        path = rt_chunks.write_abort_report(out_dir, dataset, stage, exc.reason)
        logger.error(f"ABORT {dataset}/stage{stage}: {exc.reason} (report: {path}). Later stages are NOT run.")
        return rt_config.EXIT_ABORT
    for s in summaries:
        if not s.get("skipped"):
            print(f"{s['dataset']}/stage{s['stage']}: wall={s['wall_seconds']:.0f}s peak_rss={s['peak_rss_mb']:.0f}MB "
                  f"chunks computed/skipped={s['chunks_computed']}/{s['chunks_skipped']}")
    logger.info("RT_RUN COMPLETE")
    return 0

if __name__ == "__main__":
    sys.exit(main())
