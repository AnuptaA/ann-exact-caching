#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Follow-up B: perturbation-size sweep for stage 4 (real database, locality queries).

For each maximum perturbation angle X (degrees) the stage 4 construction is repeated with the same 50 anchors,
50 train and 50 test queries per anchor, real database, labeled set = S sampled vectors plus the train queries.
Aggregates only (no per-query rows, no dump). Seeds 0 to 2, S = 10,000, nu in {1, 8, 32}, all 7 configs.

Run from the thesis root (prefer rt_followup_supervise.py):
    python3 .../rt_followup_b.py --datasets sift --xs 40
    python3 .../rt_followup_b.py --datasets both --xs 2,5,10,20,40
    python3 .../rt_followup_b.py --summary"""

import argparse
import gc
import json
import shutil
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
from data_prep import load_official_query_pool
from logging_utils import setup_logging
from perturbation import random_tangent_direction, tangent_perturb

FOLLOWUP_DIR = rt_config.RESULTS_DIR / "followup"
for _d in (FOLLOWUP_DIR, rt_config.PLOTS_DIR / "followup"):
    _d.mkdir(parents=True, exist_ok=True)

X_LIST = [2, 5, 10, 20, 40]
SEEDS = [0, 1, 2]
NUS = [1, 8, 32]
REF_S = rt_config.REF_S
STAGE = 4
POP_NAMES = {"sample_train": "sample+train", "sample_only": "sample-only"}
SUM_COLS = ["n", "sum_theta", "sum_rhat", "sum_theta2", "sum_rhat2", "sum_theta_rhat", "sum_relerr", "sum_relerr2",
            "n_distinct", "sum_distinct_total", "sum_distinct_train", "sum_distinct_sample", "sum_train_share"]
AGG_KEYS = ["config", "L", "m", "S", "pop", "nu", "seed", "distinct_scope"]

#-------------------------------------------------------------------------------
# Locality construction with a parameterized maximum angle (same rng stream as rt_data._build_locality)

def build_locality_x(pool: np.ndarray, database: np.ndarray, n_train: int, n_test: int, max_angle_deg: float,
                     logger, label: str) -> dict:
    rng = np.random.default_rng([rt_config.DATA_SEED, 300])
    n_anchors = rt_config.N_ANCHORS
    anchor_src = rng.choice(len(pool), n_anchors, replace=False)
    anchors = pool[anchor_src]
    max_dist = float(max_angle_deg) / 180.0
    accepted = np.empty((0, pool.shape[1]))
    sets, redraws = {}, 0
    for role, n in (("train", n_train), ("test", n_test)):
        base, extra = divmod(n, n_anchors)
        per_anchor = [base + (1 if a < extra else 0) for a in range(n_anchors)]
        q_list, a_list, p_list = [], [], []
        for a in range(n_anchors):
            need, got = per_anchor[a], 0
            while got < need:
                batch = max(8, 2 * (need - got))
                cand, real = [], []
                for _ in range(batch):
                    u = rng.uniform(0.0, max_dist)
                    q, realized = tangent_perturb(anchors[a], random_tangent_direction(anchors[a], rng), u)
                    cand.append(q)
                    real.append(realized)
                cand, real = np.array(cand), np.array(real)
                valid = rt_data.nn_distance(cand, database) > rt_config.DUP_TOL
                if len(accepted):
                    valid &= ~(np.arccos(np.clip(cand @ accepted.T, -1, 1)).min(axis=1) / np.pi <= rt_config.DUP_TOL)
                valid &= ~rt_data.duplicate_of_earlier(cand)
                redraws += int((~valid).sum())
                keep = np.flatnonzero(valid)[:need - got]
                accepted = np.vstack([accepted, cand[keep]])
                q_list.append(cand[keep])
                a_list.append(np.full(len(keep), a))
                p_list.append(real[keep])
                got += len(keep)
        sets[role] = (np.vstack(q_list), np.concatenate(a_list), np.concatenate(p_list))
    assert set(sets["train"][1].tolist()) == set(sets["test"][1].tolist()) == set(range(n_anchors)), \
        f"{label}: train and test must share the same {n_anchors} anchors"
    degrees = np.concatenate([sets["train"][2], sets["test"][2]]) * 180.0
    assert degrees.max() <= max_angle_deg + 1e-6
    logger.info(f"{label}: locality X={max_angle_deg}, {n_anchors} anchors, realized degrees min {degrees.min():.3f} "
                f"mean {degrees.mean():.3f} max {degrees.max():.3f}; {redraws} invalid draws redrawn")
    out = dict(train=sets["train"][0], test=sets["test"][0], degrees=degrees, redraws=redraws)
    for role in ("train", "test"):
        out[f"meta_{role}"] = rt_data._meta(role, len(sets[role][0]), anchor_id=sets[role][1],
                                            pert_dist=sets[role][2], source=anchor_src[sets[role][1]])
    return out

#-------------------------------------------------------------------------------
# Reduced state: the S = REF_S population only (same columns as the S = REF_S population of the full state)

def build_reduced_state(seed: int, database: np.ndarray, theta_db: np.ndarray, train: np.ndarray,
                        theta_train: np.ndarray, S: int = REF_S) -> rt_engine.SeedState:
    perm = rt_data.sample_permutation(len(database), seed)[:S]
    W = rt_engine.make_hyperplanes(seed, database.shape[1])
    codes = np.concatenate([rt_engine.sign_codes(database[perm], W), rt_engine.sign_codes(train, W)])
    theta_lab = np.concatenate([theta_db[perm], theta_train]).astype(np.float64)
    return rt_engine.SeedState(seed, S, len(train), W, perm, codes, theta_lab, [S], [0, S, S + len(train)])

#-------------------------------------------------------------------------------
# Aggregate-only chunk store

class AggChunkStore(rt_chunks.ChunkStore):
    def is_valid(self, seed: int, chunk: int, expected_agg: int) -> bool:
        dp = self.done_path(seed, chunk)
        if not dp.exists():
            return False
        try:
            info = json.loads(dp.read_text())
            return info["agg"] == expected_agg == len(rt_chunks.read_table(self.name(seed, chunk, "agg.csv.gz")))
        except Exception:
            return False

    def write_agg(self, seed: int, chunk: int, agg: pd.DataFrame) -> None:
        import os
        rt_chunks.write_table_atomic(agg, self.name(seed, chunk, "agg.csv.gz"))
        tmp = self.done_path(seed, chunk).with_name("done.tmp")
        tmp.write_text(json.dumps(dict(agg=len(agg))))
        os.replace(tmp, self.done_path(seed, chunk))

#-------------------------------------------------------------------------------

def run_point(dataset: str, X: float, pool: np.ndarray, database: np.ndarray, theta_db: np.ndarray, seeds: list,
              chunk_log: rt_chunks.ChunkLog, logger, rss_limit_mb: float, min_free_disk_gb: float,
              chunk_size: int = rt_config.CHUNK_SIZE, out_dir: Path = FOLLOWUP_DIR) -> dict:
    t_all = time.time()
    label = f"{dataset}/stage4/X{X:g}"
    free = rt_chunks.free_disk_gb(out_dir)
    if free < min_free_disk_gb:
        raise rt_chunks.RunAbort(f"free disk {free:.2f} GiB below {min_free_disk_gb} GiB at {label} start")
    n_train, n_test = rt_config.N_TRAIN, rt_config.N_TEST
    timings = {}
    t0 = time.time()
    qs = build_locality_x(pool, database, n_train, n_test, X, logger, label)
    qs.update(database=database)
    rt_data.check_query_sets(qs, logger, label)
    timings["build_queries"] = time.time() - t0

    t0 = time.time()
    cov = rt_exact.query_covariates(np.vstack([qs["train"], qs["test"]]), database)
    assert cov["theta"].gt(0).all() and cov["nn1"].gt(rt_config.DUP_TOL).all()
    theta_train, theta_test = cov["theta"].to_numpy()[:n_train], cov["theta"].to_numpy()[n_train:]
    timings["exact_pass"] = time.time() - t0
    logger.info(f"{label}: exact pass {timings['exact_pass']:.0f}s, test theta mean {theta_test.mean():.4f} "
                f"(= {theta_test.mean() * 180:.1f} degrees), relative sd {100 * theta_test.std() / theta_test.mean():.2f}%")

    import hashlib
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(qs["train"]).tobytes())
    h.update(np.ascontiguousarray(qs["test"]).tobytes())
    fp = rt_chunks.run_fingerprint(dataset, STAGE, seeds, n_train, n_test, rt_config.CONFIGS, NUS, [REF_S], chunk_size,
                                   len(database), f"X{X:g}:{h.hexdigest()}")
    stem = out_dir / f"{dataset}_stage4_X{X:g}"
    if rt_chunks.stage_complete(stem, fp):
        logger.info(f"{label}: already complete ({fp}), skipping")
        return dict(skipped=True)
    store = AggChunkStore(out_dir, dataset, STAGE, fp)

    opts = rt_engine.ChunkOptions(nu_grid=list(NUS), ref_S=REF_S, union_all_seeds=[REF_S])
    n_cfg = len(rt_config.CONFIGS)
    n_agg = n_cfg * len(rt_config.POP_KINDS) * len(NUS)
    states, plan, peak, phases, computed, skipped = {}, [], 0.0, {}, 0, 0
    n_chunks = int(np.ceil(n_test / chunk_size))
    for ci in range(n_chunks):
        s, e = ci * chunk_size, min(n_test, (ci + 1) * chunk_size)
        for seed in seeds:
            plan.append((seed, ci))
            if store.is_valid(seed, ci, n_agg):
                skipped += 1
                logger.info(chunk_log.write(dataset, STAGE, seed, ci, f"SKIP_X{X:g}", 0.0, rt_chunks.current_rss_mb()))
                continue
            if seed not in states:
                tb = time.time()
                states[seed] = build_reduced_state(seed, database, theta_db, qs["train"], theta_train)
                timings["seed_state"] = timings.get("seed_state", 0.0) + time.time() - tb
            t0 = time.time()
            out = rt_engine.process_chunk(states[seed], qs["test"][s:e], theta_test[s:e], np.arange(s, e), opts)
            reason = rt_chunks.integrity_reason(out["rows"], out["agg"])
            if reason:
                raise rt_chunks.RunAbort(f"{label} seed {seed} chunk {ci}: {reason} (chunk NOT written)")
            assert len(out["agg"]) == n_agg
            store.write_agg(seed, ci, out["agg"])
            elapsed = time.time() - t0
            rss = rt_chunks.current_rss_mb()
            peak = max(peak, rss)
            computed += 1
            for k, v in out["timings"].items():
                phases[k] = phases.get(k, 0.0) + v
            logger.info(chunk_log.write(dataset, STAGE, seed, ci, f"DONE_X{X:g}", elapsed, rss, out["timings"]))
            del out
            gc.collect()
            if rss > rss_limit_mb:
                raise rt_chunks.RunAbort(f"RSS {rss:.0f} MB exceeds {rss_limit_mb:.0f} MB after {label} seed {seed} "
                                         f"chunk {ci} (chunk was written and flushed)")
            free = rt_chunks.free_disk_gb(out_dir)
            if free < min_free_disk_gb:
                raise rt_chunks.RunAbort(f"free disk {free:.2f} GiB below {min_free_disk_gb} GiB after {label}")

    agg = pd.concat([rt_chunks.read_table(store.name(s, c, "agg.csv.gz")) for s, c in plan], ignore_index=True)
    sums = [c for c in agg.columns if c not in AGG_KEYS]
    agg = agg.groupby(AGG_KEYS, sort=False)[sums].sum(min_count=1).reset_index()
    assert agg[["n", "sum_theta", "sum_rhat"]].notna().all().all()
    assert (agg["n"] == n_test).all(), f"{label}: aggregate n differs from the number of test queries"
    assert np.allclose(agg["sum_theta"] / agg["n"], theta_test.mean(), rtol=1e-12), f"{label}: theta sums disagree"
    # EXACT estimate is r_hat = theta by definition: its error is exactly 0
    exact_err = float(np.abs(theta_test.copy() - theta_test).max())
    if exact_err != 0.0:
        raise rt_chunks.RunAbort(f"{label}: EXACT error {exact_err} is not 0")
    agg.insert(0, "X", float(X))
    agg.insert(0, "dataset", dataset)
    rt_chunks.write_table_atomic(agg, Path(f"{stem}_aggregates.csv.gz"))
    summary = dict(dataset=dataset, X=float(X), fingerprint=fp, seeds=list(seeds), S=REF_S, nus=NUS,
                   test_theta_mean=float(theta_test.mean()), test_theta_mean_degrees=float(theta_test.mean() * 180),
                   test_theta_relative_sd=float(theta_test.std() / theta_test.mean()),
                   realized_degrees=dict(min=float(qs["degrees"].min()), mean=float(qs["degrees"].mean()),
                                         max=float(qs["degrees"].max())),
                   redraws=int(qs["redraws"]), exact_max_abs_error=exact_err, timings_seconds=timings,
                   chunk_phase_seconds=phases, chunks_computed=computed, chunks_skipped=skipped,
                   peak_rss_mb=peak, wall_seconds=time.time() - t_all, finished=datetime.now().isoformat())
    Path(f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))
    rt_chunks.marker_path(stem).write_text(json.dumps(dict(fingerprint=fp, written=datetime.now().isoformat())))
    shutil.rmtree(store.dir, ignore_errors=True)
    logger.info(f"{label}: COMPLETE in {summary['wall_seconds']:.0f}s (peak RSS {peak:.0f} MB)")
    return summary

#-------------------------------------------------------------------------------
# Summary tables from saved sums

def metrics_from_sums(agg: pd.DataFrame) -> pd.DataFrame:
    """Per aggregate row: mean error, sd of error, forward/reverse slope, R squared, mean distinct, train share."""
    a = agg.copy()
    n = a["n"]
    mx, my = a["sum_theta"] / n, a["sum_rhat"] / n
    cxx = a["sum_theta2"] / n - mx ** 2
    cyy = a["sum_rhat2"] / n - my ** 2
    cxy = a["sum_theta_rhat"] / n - mx * my
    a["slope"] = cxy / cxx
    a["reverse_slope"] = cxy / cyy
    a["r_squared"] = cxy ** 2 / (cxx * cyy)
    a["mean_error_pct"] = 100 * a["sum_relerr"] / n
    a["sd_error_pct"] = 100 * np.sqrt(np.maximum(a["sum_relerr2"] / n - (a["sum_relerr"] / n) ** 2, 0))
    a["mean_distinct"] = a["sum_distinct_total"] / a["n_distinct"]
    a["train_share"] = a["sum_train_share"] / a["n_distinct"]
    a["estimate"] = a["pop"].map(POP_NAMES)
    return a

def build_summary(results_dir: Path = FOLLOWUP_DIR) -> pd.DataFrame:
    parts = [rt_chunks.read_table(p) for p in sorted(results_dir.glob("*_stage4_X*_aggregates.csv.gz"))]
    agg = pd.concat(parts, ignore_index=True)
    m = metrics_from_sums(agg)
    cols = ["mean_error_pct", "sd_error_pct", "slope", "reverse_slope", "r_squared", "mean_distinct", "train_share"]
    keys = ["dataset", "X", "config", "nu", "estimate"]
    g = m.groupby(keys, sort=False)
    out = g[cols].mean()
    for c in cols:
        out[f"{c}_sd"] = g[c].std(ddof=1)
    out["n_seeds"] = g.size()
    out = out.reset_index()
    out.to_csv(results_dir / "B_summary.csv", index=False)
    return out

def sanity_check_x10(results_dir: Path = FOLLOWUP_DIR, main_dir: Path = rt_config.RESULTS_DIR) -> dict:
    """Sweep aggregates at X = 10 versus the saved stage 4 aggregates (seeds 0 to 2, S = 10,000, nu in NUS)."""
    report = {}
    for ds in rt_config.DATASETS:
        p = results_dir / f"{ds}_stage4_X10_aggregates.csv.gz"
        if not p.exists():
            continue
        new = rt_chunks.read_table(p)
        old = rt_chunks.read_table(main_dir / f"{ds}_stage4_aggregates.csv.gz")
        old = old[(old["S"] == REF_S) & old["nu"].isin(NUS) & old["seed"].isin(SEEDS) &
                  old["config"].isin(rt_config.CONFIG_NAMES)]
        keys = ["config", "pop", "nu", "seed"]
        mg = new.merge(old, on=keys, suffixes=("_new", "_old"))
        worst_abs, worst_rel, ok = 0.0, 0.0, True
        for c in SUM_COLS:
            a, b = mg[f"{c}_new"].to_numpy(float), mg[f"{c}_old"].to_numpy(float)
            worst_abs = max(worst_abs, float(np.nanmax(np.abs(a - b))))
            denom = np.maximum(np.abs(b), 1e-300)
            worst_rel = max(worst_rel, float(np.nanmax(np.abs(a - b) / denom)))
            ok &= bool(np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True))
        report[ds] = dict(rows_compared=int(len(mg)), rows_new=int(len(new)), rows_old=int(len(old)),
                          allclose_rtol1e9=ok, max_abs_diff=worst_abs, max_rel_diff=worst_rel)
    (results_dir / "B_sanity_X10.json").write_text(json.dumps(report, indent=2))
    return report

#-------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="both", choices=["sift", "esci", "both"])
    ap.add_argument("--xs", default=",".join(str(x) for x in X_LIST))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--summary", action="store_true", help="only build B_summary.csv and the X = 10 sanity check")
    ap.add_argument("--rss-limit-mb", type=float, default=rt_config.RSS_LIMIT_MB)
    ap.add_argument("--min-free-disk-gb", type=float, default=rt_config.MIN_FREE_DISK_GB)
    args = ap.parse_args()
    if args.summary:
        s = build_summary()
        print(json.dumps(sanity_check_x10(), indent=2))
        print(f"B_summary.csv rows: {len(s)}")
        return 0
    datasets = rt_config.DATASETS if args.datasets == "both" else [args.datasets]
    xs = [float(x) for x in args.xs.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    logger = setup_logging("rt_followup_b", rt_config.LOGS_DIR)
    chunk_log = rt_chunks.ChunkLog(rt_config.LOGS_DIR / "chunk_log_followup.csv")
    logger.info(f"RT_FOLLOWUP_B STARTING datasets={datasets} X={xs} seeds={seeds} S={REF_S} nu={NUS}")
    summaries, ds_cur, x_cur = [], None, None
    try:
        for ds in datasets:
            ds_cur = ds
            pool = load_official_query_pool(ds)
            database = rt_data.build_database(ds, STAGE)
            theta_db, _ = rt_data.build_or_load_theta(rt_config.stage_db_key(ds, STAGE), database, logger)
            for X in xs:
                x_cur = X
                summaries.append(run_point(ds, X, pool, database, theta_db, seeds, chunk_log, logger,
                                           args.rss_limit_mb, args.min_free_disk_gb))
            del pool
    except rt_chunks.RunAbort as exc:
        path = rt_chunks.write_abort_report(FOLLOWUP_DIR, f"{ds_cur}_followupB_X{x_cur:g}", STAGE, exc.reason)
        logger.error(f"ABORT {ds_cur}/X{x_cur}: {exc.reason} (report: {path}). Remaining points are NOT run.")
        return rt_config.EXIT_ABORT
    for s in summaries:
        if not s.get("skipped"):
            print(f"{s['dataset']}/X{s['X']:g}: wall={s['wall_seconds']:.0f}s peak_rss={s['peak_rss_mb']:.0f}MB "
                  f"theta_mean_deg={s['test_theta_mean_degrees']:.1f} rel_sd={100 * s['test_theta_relative_sd']:.1f}%")
    logger.info("RT_FOLLOWUP_B COMPLETE")
    return 0

if __name__ == "__main__":
    sys.exit(main())
