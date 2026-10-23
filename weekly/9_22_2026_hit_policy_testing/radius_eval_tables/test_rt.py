#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Tests for radius_eval_tables. Run with pytest or `python3 -B test_rt.py` (writes only to temp dirs)."""

import inspect
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import rt_chunks
import rt_config
import rt_data
import rt_engine
import rt_exact
import rt_run
from ground_truth import batched_topk_full_skip_self

LOG = logging.getLogger("test_rt")

#-------------------------------------------------------------------------------
# Small fixture: N=400 database, 60 train, bits_max 256, S grid (100, 200, 400)

BITS = 256
S_GRID = [100, 200, 400]
REF = 200
CONFIGS = [(1, 128), (2, 64), (1, 256), (4, 64), (2, 128)]
NU = [1, 2, 4, 8, 16, 32]

def make_fixture(seed: int = 0, N: int = 400, d: int = 16, n_train: int = 60, n_test: int = 30):
    rng = np.random.default_rng(100 + seed)
    db = rt_data.sample_uniform_sphere(N, d, rng)
    train = rt_data.sample_uniform_sphere(n_train, d, rng)
    test = rt_data.sample_uniform_sphere(n_test, d, rng)
    theta_db = rng.uniform(0.3, 0.7, N)
    theta_train = rng.uniform(0.3, 0.7, n_train)
    theta_test = rng.uniform(0.3, 0.7, n_test)
    state = rt_engine.build_seed_state(seed, db, theta_db, train, theta_train, S_GRID, BITS)
    return dict(db=db, train=train, test=test, theta_db=theta_db, theta_train=theta_train, theta_test=theta_test,
                state=state)

def direct_hamming(state, q_codes, lo, m):
    return np.array([(state.codes[:, lo:lo + m] != qc[None, lo:lo + m]).sum(axis=1) for qc in q_codes])

def pop_cols(state, S, kind):
    cols = np.arange(S)
    return cols if kind == "sample_only" else np.concatenate([cols, np.arange(state.N, state.n_labeled)])

def brute_table(h_row, theta_cols, nu):
    cut = np.sort(h_row)[nu - 1]
    sel = h_row <= cut
    return theta_cols[sel].mean(), cut, sel

def brute_config(state, q_codes, r, L, m, S, kind, nu):
    cols = pop_cols(state, S, kind)
    vals, cuts, union = [], [], np.zeros(len(cols), bool)
    for lo, mm in rt_config.config_tables(L, m):
        h = (state.codes[cols, lo:lo + mm] != q_codes[r, lo:lo + mm][None, :]).sum(axis=1)
        v, c, sel = brute_table(h, state.theta_lab[cols], nu)
        vals.append(v)
        cuts.append(c)
        union |= sel
    return np.mean(vals), vals, cuts, cols, union

def run_chunk(fx, opts=None):
    opts = opts or rt_engine.ChunkOptions(configs=CONFIGS, ref_S=REF)
    return rt_engine.process_chunk(fx["state"], fx["test"], fx["theta_test"], np.arange(len(fx["test"])), opts)

#-------------------------------------------------------------------------------

def test_nested_sample_prefix():
    p = rt_data.sample_permutation(1000, 3)
    assert np.array_equal(p[:100], rt_data.sample_permutation(1000, 3)[:100])
    fx = make_fixture()
    assert np.array_equal(fx["state"].perm[:100], fx["state"].perm[:200][:100])
    assert np.allclose(fx["state"].theta_lab[:100], fx["theta_db"][fx["state"].perm[:100]])
    assert not np.array_equal(rt_data.sample_permutation(1000, 3), rt_data.sample_permutation(1000, 4))

def test_table_rows_and_single_long_code():
    assert rt_config.config_tables(4, 128) == [(0, 128), (128, 128), (256, 128), (384, 128)]
    assert rt_config.config_tables(1, 512) == [(0, 512)]
    d = 12
    rng = np.random.default_rng(1)
    db = rt_data.sample_uniform_sphere(30, d, rng)
    tr = rt_data.sample_uniform_sphere(6, d, rng)
    q = rt_data.sample_uniform_sphere(5, d, rng)
    state = rt_engine.build_seed_state(0, db, np.ones(30), tr, np.ones(6), [10, 30], 512)
    qc = rt_engine.sign_codes(q, state.W)
    H = rt_engine.table_hamming(qc.astype(np.float32), state, [(0, 512), (0, 128)])
    assert np.array_equal(H[(0, 512)], direct_hamming(state, qc, 0, 512))
    first4 = sum(rt_engine.table_hamming(qc.astype(np.float32), state, [(128 * t, 128)])[(128 * t, 128)].astype(int)
                 for t in range(4))
    assert np.array_equal(first4, H[(0, 512)])

def test_hamming_matches_direct_bit_count_including_m2048():
    d = 10
    rng = np.random.default_rng(2)
    db = rt_data.sample_uniform_sphere(40, d, rng)
    tr = rt_data.sample_uniform_sphere(8, d, rng)
    q = rt_data.sample_uniform_sphere(6, d, rng)
    q[0] = -db[7]  # antipodal to a labeled vector: distance 2048 > uint8 range
    state = rt_engine.build_seed_state(1, db, np.ones(40), tr, np.ones(8), [10, 40], 2048)
    qc = rt_engine.sign_codes(q, state.W)
    tables = rt_config.unique_tables()
    H = rt_engine.table_hamming(qc.astype(np.float32), state, tables)
    for (lo, m), arr in H.items():
        assert arr.dtype == np.uint16
        assert np.array_equal(arr, direct_hamming(state, qc, lo, m)), (lo, m)
    col = int(np.flatnonzero(state.perm == 7)[0])
    assert H[(0, 2048)][0, col] == 2048

def test_histogram_path_matches_brute_force_both_populations():
    fx = make_fixture(5)
    st = fx["state"]
    out = run_chunk(fx)
    q_codes = rt_engine.sign_codes(fx["test"], st.W)
    rows = out["rows"]
    ties_seen = False
    for L, m in CONFIGS:
        cname = rt_config.config_name(L, m)
        for nu in NU:
            sub = rows[(rows["config"] == cname) & (rows["nu"] == nu)].sort_values("query_id")
            for kind, col in (("sample_train", "r_hat"), ("sample_only", "r_hat_sample_only")):
                ref = np.array([brute_config(st, q_codes, r, L, m, REF, kind, nu)[0] for r in range(len(q_codes))])
                assert np.allclose(sub[col].to_numpy(), ref, rtol=1e-6, atol=1e-7), (cname, nu, kind)
            for r in range(len(q_codes)):
                _, vals, cuts, cols, union = brute_config(st, q_codes, r, L, m, REF, "sample_train", nu)
                h = (st.codes[cols, 0:m] != q_codes[r, 0:m][None, :]).sum(axis=1)
                ties_seen |= (h <= np.sort(h)[nu - 1]).sum() > nu
    assert ties_seen, "fixture must exercise ties"

def test_aggregates_cover_all_populations_and_match_brute_force():
    fx = make_fixture(6)
    st = fx["state"]
    out = run_chunk(fx)
    q_codes = rt_engine.sign_codes(fx["test"], st.W)
    agg = out["agg"]
    for S in S_GRID:
        for kind in rt_config.POP_KINDS:
            for L, m in CONFIGS[:3]:
                for nu in (1, 8):
                    row = agg[(agg["S"] == S) & (agg["pop"] == kind) & (agg["config"] == rt_config.config_name(L, m))
                              & (agg["nu"] == nu)].iloc[0]
                    ref = np.array([brute_config(st, q_codes, r, L, m, S, kind, nu)[0] for r in range(len(q_codes))])
                    assert np.isclose(row["sum_rhat"], ref.sum(), rtol=1e-9)
                    assert np.isclose(row["sum_relerr"], ((ref - fx["theta_test"]) / fx["theta_test"]).sum(), rtol=1e-8)
    assert len(agg) == len(CONFIGS) * len(S_GRID) * 2 * len(NU)

def test_r_hat_is_plain_mean_of_table_means():
    fx = make_fixture(7)
    st = fx["state"]
    out = run_chunk(fx)
    q_codes = rt_engine.sign_codes(fx["test"], st.W)
    rows = out["rows"]
    L, m, nu = 4, 64, 4
    sub = rows[(rows["config"] == "L4_m64") & (rows["nu"] == nu)].sort_values("query_id")
    for r in range(5):
        _, vals, _, _, _ = brute_config(st, q_codes, r, L, m, REF, "sample_train", nu)
        assert np.isclose(sub["r_hat"].iloc[r], np.mean(vals), rtol=1e-6)
        spread = np.std(vals) / np.mean(vals)
        assert np.isclose(sub["table_spread"].iloc[r], spread, rtol=1e-4, atol=1e-6)
    single = rows[rows["config"] == "L1_m128"]
    assert single["table_spread"].isna().all()

def test_sample_only_excludes_train_points():
    fx = make_fixture(8)
    fx["theta_train"][:] = 1000.0
    st = rt_engine.build_seed_state(0, fx["db"], fx["theta_db"], fx["train"], fx["theta_train"], S_GRID, BITS)
    fx["state"] = st
    rows = run_chunk(fx)["rows"]
    assert (rows["r_hat_sample_only"] <= 1.0).all()
    assert (rows["r_hat"] > 1.0).any()
    agg = run_chunk(fx)["agg"]
    so = agg[agg["pop"] == "sample_only"]
    assert (so["sum_distinct_train"].dropna() == 0).all()

def test_group_means_flags_and_union_counts():
    fx = make_fixture(9)
    st = fx["state"]
    rows = run_chunk(fx)["rows"]
    q_codes = rt_engine.sign_codes(fx["test"], st.W)
    ids = st.labeled_ids()
    for L, m in ((4, 64), (2, 128)):
        cname = rt_config.config_name(L, m)
        for nu in (2, 16):
            sub = rows[(rows["config"] == cname) & (rows["nu"] == nu)].sort_values("query_id")
            for r in range(6):
                _, _, cuts, cols, union = brute_config(st, q_codes, r, L, m, REF, "sample_train", nu)
                members = cols[union]
                train_members = members[members >= st.N]
                sample_members = members[members < st.N]
                assert sub["n_distinct_total"].iloc[r] == len(members)
                assert sub["n_distinct_train"].iloc[r] == len(train_members)
                assert sub["n_distinct_sample"].iloc[r] == len(sample_members)
                if len(train_members):
                    assert np.isclose(sub["mean_theta_train_group"].iloc[r], st.theta_lab[train_members].mean(),
                                      rtol=1e-6)
                else:
                    assert np.isnan(sub["mean_theta_train_group"].iloc[r])
                assert np.isclose(sub["mean_theta_sample_group"].iloc[r], st.theta_lab[sample_members].mean(), rtol=1e-6)
    assert (ids[st.N:] == st.N + np.arange(st.n_train)).all()

def test_union_scope_flag_seed0_only_for_other_S():
    fx0, fx1 = make_fixture(0), make_fixture(1)
    opts = rt_engine.ChunkOptions(configs=CONFIGS[:2], ref_S=REF, union_all_seeds=[REF])
    a0, a1 = run_chunk(fx0, opts)["agg"], run_chunk(fx1, opts)["agg"]
    assert a0[a0["S"] != REF]["sum_distinct_total"].notna().all()
    assert a1[a1["S"] != REF]["sum_distinct_total"].isna().all()
    assert a1[a1["S"] == REF]["sum_distinct_total"].notna().all()
    assert set(a0[a0["S"] != REF]["distinct_scope"]) == {"seed0_only"}
    assert set(a0[a0["S"] == REF]["distinct_scope"]) == {"all_seeds"}

def test_query_sets_disjoint_and_checked():
    logger = LOG
    rng = np.random.default_rng(3)
    db = rt_data.sample_uniform_sphere(300, 16, rng)
    cand = rt_data.sample_uniform_sphere(130, 16, rng)
    cand[10] = db[5]          # equals a database vector
    cand[20] = cand[19]       # duplicate query
    out = rt_data._split_plain(cand, np.arange(130), db, 50, 50, logger, "t")
    out.update(database=db)
    rt_data.check_query_sets(out, logger, "t")
    both = np.vstack([out["train"], out["test"]])
    assert len(np.unique(both, axis=0)) == 100
    bad = dict(out)
    bad["test"] = bad["test"].copy()
    bad["test"][0] = bad["train"][0]
    try:
        rt_data.check_query_sets(bad, logger, "bad")
        raise RuntimeError("check_query_sets must reject a test query equal to a train query")
    except AssertionError:
        pass
    bad2 = dict(out)
    bad2["test"] = bad2["test"].copy()
    bad2["test"][1] = db[3]
    try:
        rt_data.check_query_sets(bad2, logger, "bad2")
        raise RuntimeError("check_query_sets must reject a test query equal to a database vector")
    except AssertionError:
        pass

def test_locality_angles_independent_and_shared_anchors():
    rng = np.random.default_rng(4)
    pool = rt_data.sample_uniform_sphere(200, 16, rng)
    db = rt_data.sample_uniform_sphere(300, 16, rng)
    qs = rt_data._build_locality(pool, db, 100, 100, LOG, "t")
    for role in ("train", "test"):
        meta = qs[f"meta_{role}"]
        Q = qs[role]
        anchors = pool[meta["source_index"].to_numpy()]
        ang = np.degrees(np.arccos(np.clip((Q * anchors).sum(axis=1), -1, 1)))
        assert ang.min() >= 0.0 and ang.max() <= 10.0 + 1e-6
        assert np.allclose(ang, meta["pert_dist"].to_numpy() * 180.0, atol=1e-5)
        assert meta["anchor_id"].nunique() == 50
        assert (meta.groupby("anchor_id").size() == 2).all()
    assert set(qs["meta_train"]["anchor_id"]) == set(qs["meta_test"]["anchor_id"])
    assert np.array_equal(qs["meta_train"].groupby("anchor_id")["source_index"].first().to_numpy(),
                          qs["meta_test"].groupby("anchor_id")["source_index"].first().to_numpy())
    both = np.vstack([qs["train"], qs["test"]])
    assert len(np.unique(both, axis=0)) == len(both)
    assert len(np.unique(qs["meta_train"]["pert_dist"])) > 90  # independent draws, not shared angles

def test_uniform_sampler_unit_norm():
    v = rt_data.sample_uniform_sphere(500, 24, np.random.default_rng(0))
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-12)

def test_theta_of_database_vector_excludes_itself():
    rng = np.random.default_rng(0)
    db = rt_data.sample_uniform_sphere(60, 8, rng)
    _, dist = batched_topk_full_skip_self(db, db, np.arange(60), 5, batch_size=16)
    full = np.arccos(np.clip(db @ db.T, -1, 1)) / np.pi
    np.fill_diagonal(full, np.inf)
    assert np.allclose(dist[:, 4], np.sort(full, axis=1)[:, 4], atol=1e-12)
    assert (dist[:, 0] > 0).all()

def test_covariates_match_sorted_distances():
    rng = np.random.default_rng(1)
    db = rt_data.sample_uniform_sphere(300, 8, rng)
    q = rt_data.sample_uniform_sphere(20, 8, rng)
    cov = rt_exact.query_covariates(q, db, K=100, chunk_size=7)
    srt = np.sort(np.arccos(np.clip(q @ db.T, -1, 1)) / np.pi, axis=1)
    assert np.allclose(cov["theta"], srt[:, 99]) and np.allclose(cov["nn1"], srt[:, 0])
    assert np.allclose(cov["d10"], srt[:, 9]) and np.allclose(cov["d50"], srt[:, 49])

def test_predict_signature_takes_only_queries():
    sig = inspect.signature(rt_engine.MultiTableEstimator.predict_radius)
    assert list(sig.parameters) == ["self", "Q"]
    fx = make_fixture(2)
    est = rt_engine.MultiTableEstimator(fx["state"], CONFIGS, NU, REF)
    rows = est.predict_radius(fx["test"])
    ref = run_chunk(fx)["rows"]
    assert np.allclose(rows["r_hat"], ref["r_hat"])

def test_dump_reproduces_estimate_and_end_to_end_aggregates():
    fx = make_fixture(0)
    st = fx["state"]
    n_test = len(fx["test"])
    chunk = 12
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        store = rt_chunks.ChunkStore(td, "sift", 3, "fp")
        plan = []
        dump_ids = np.arange(0, n_test, 3)
        for seed in (0,):
            for ci in range(int(np.ceil(n_test / chunk))):
                s, e = ci * chunk, min(n_test, (ci + 1) * chunk)
                sel = np.flatnonzero((dump_ids >= s) & (dump_ids < e))
                opts = rt_engine.ChunkOptions(configs=CONFIGS, ref_S=REF, dump_local=dump_ids[sel] - s)
                out = rt_engine.process_chunk(st, fx["test"][s:e], fx["theta_test"][s:e], np.arange(s, e), opts)
                assert rt_chunks.integrity_reason(out["rows"], out["agg"]) is None
                store.write_chunk(seed, ci, out["rows"], out["agg"], out["dump"])
                plan.append((seed, ci))
        cov = pd.DataFrame(dict(role=["train"] * st.n_train + ["test"] * n_test,
                                query_id=list(range(st.n_train)) + list(range(n_test)),
                                theta=np.concatenate([fx["theta_train"], fx["theta_test"]])))
        summary = dict(N=st.N)
        rt_run.consolidate(store, td / "t", plan, cov, fx["theta_db"], [0], True, "fp", LOG, summary, S_GRID, REF)
        rows = rt_chunks.read_table(td / "t_per_query_ref.csv.gz")
        agg = rt_chunks.read_table(td / "t_aggregates.csv.gz")
        dump = rt_chunks.read_table(td / "t_neighbor_dump.csv.gz")
        assert (td / "t.ok").exists() and not store.dir.exists()
        ex = rows[rows["config"] == "EXACT"]
        assert (ex["r_hat"].to_numpy() == fx["theta_test"]).all()
        a_ex = agg[agg["config"] == "EXACT"].iloc[0]
        assert a_ex["sum_relerr"] == 0.0 and a_ex["sum_rhat"] == a_ex["sum_theta"]

        theta = fx["theta_test"]
        for cname in ("L1_m128", "L4_m64"):
            for nu in (1, 8):
                r = rows[(rows["config"] == cname) & (rows["nu"] == nu)].sort_values("query_id")["r_hat"].to_numpy()
                row = agg[(agg["config"] == cname) & (agg["S"] == REF) & (agg["pop"] == "sample_train")
                          & (agg["nu"] == nu) & (agg["seed"] == 0)].iloc[0]
                n = row["n"]
                slope_agg = (n * row["sum_theta_rhat"] - row["sum_theta"] * row["sum_rhat"]) / (
                    n * row["sum_theta2"] - row["sum_theta"] ** 2)
                slope_rows = np.polyfit(theta, r, 1)[0]
                assert np.isclose(slope_agg, slope_rows, rtol=1e-4)
                assert np.isclose(row["sum_relerr"] / n, np.mean((r - theta) / theta), rtol=1e-4, atol=1e-6)

        # dump reproduces the estimate when no table's neighborhood is truncated by the tie cap
        checked = 0
        dump_q = sorted(dump["query_id"].unique())
        for cname, (L, m) in ((rt_config.config_name(*c), c) for c in CONFIGS):
            for nu in NU:
                sub = rows[(rows["config"] == cname) & (rows["nu"] == nu)].set_index("query_id")
                for q in dump_q:
                    vals, fits = [], True
                    for t in range(L):
                        g = dump[(dump["query_id"] == q) & (dump["config"] == cname) & (dump["table"] == t)] \
                            .sort_values("rank_in_table")
                        if len(g) < nu:
                            fits = False
                            break
                        h_nu = g["hamming"].iloc[nu - 1]
                        sel = g[g["hamming"] <= h_nu]
                        cut_full = g["hamming"].iloc[-1]
                        if h_nu == cut_full and len(g) >= rt_config.DUMP_TIE_CAP:
                            fits = False
                            break
                        vals.append(sel["theta_neighbor"].astype(np.float64).mean())
                    if fits:
                        checked += 1
                        assert np.isclose(np.mean(vals), sub.loc[q, "r_hat"], rtol=1e-4), (cname, nu, q)
        assert checked > 100
        ids = dump["labeled_id"].to_numpy()
        assert ((dump["from_train"].to_numpy() == 1) == (ids >= st.N)).all()
        assert dump.groupby(["query_id", "config", "table"]).size().max() <= rt_config.DUMP_TIE_CAP
        assert (dump["rank_in_table"] >= 1).all()

def test_chunk_store_validity_and_stale_tmp_removed():
    with tempfile.TemporaryDirectory() as td:
        store = rt_chunks.ChunkStore(Path(td), "sift", 0, "x")
        (store.dir / "partial.csv.gz.tmp").write_bytes(b"junk")
        store = rt_chunks.ChunkStore(Path(td), "sift", 0, "x")
        assert not list(store.dir.glob("*.tmp"))
        rows = pd.DataFrame(dict(query_id=[1, 2], x=[0.1, 0.2]))
        store.write_chunk(0, 0, rows, rows, None)
        assert store.is_valid(0, 0, 2, 2, False)
        assert not store.is_valid(0, 0, 3, 2, False)
        assert not store.is_valid(0, 1, 2, 2, False)
        store.name(0, 0, "rows.csv.gz").write_bytes(b"corrupt")
        assert not store.is_valid(0, 0, 2, 2, False)

def test_integrity_reason_flags_nan_and_duplicates():
    fx = make_fixture(1)
    out = run_chunk(fx)
    assert rt_chunks.integrity_reason(out["rows"], out["agg"]) is None
    bad = out["rows"].copy()
    bad.loc[0, "r_hat"] = np.nan
    assert "NaN" in rt_chunks.integrity_reason(bad, out["agg"])
    dup = pd.concat([out["rows"], out["rows"].iloc[:1]], ignore_index=True)
    assert "duplicate" in rt_chunks.integrity_reason(dup, out["agg"])

def test_stage0_builds_and_is_valid_small():
    qs = rt_data.build_stage("sift", 0, LOG, 20, 20)
    assert qs["database"].shape == (rt_config.N_DB, 128) and qs["train"].shape == (20, 128)
    assert np.allclose(np.linalg.norm(qs["database"], axis=1), 1.0)

#-------------------------------------------------------------------------------

def run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:
            failed += 1
            import traceback
            print(f"FAIL {name}: {exc!r}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return failed

if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)
