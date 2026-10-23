#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Tests for the report modules. Run with pytest or `python3 -B test_rt_report.py` (no files written)."""

import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import rt_config
import rt_engine
import rt_report_data as rd
import rt_report_slides as sl

HERE = Path(__file__).parent

#-------------------------------------------------------------------------------

def brute_pooled(x, y, frac=0.01):
    order = np.argsort(x, kind="stable")
    xs, ys = np.asarray(x)[order], np.asarray(y)[order]
    n = len(xs)
    h = math.ceil(0.005 * n) if frac == 0.01 else math.ceil(frac / 2 * n)
    mx, my = [], []
    for i in range(n):
        lo, hi = max(i - h, 0), min(i + h, n - 1)
        mx.append(xs[lo:hi + 1].mean())
        my.append(ys[lo:hi + 1].mean())
    return np.array(mx), np.array(my)

def test_pooled_by_nearest_pct_matches_brute_force_with_ends_and_ties():
    rng = np.random.default_rng(0)
    for n in (7, 20, 237, 1000):
        x = np.round(rng.uniform(0, 1, n), 2)  # ties
        y = rng.normal(size=n)
        px, py = rd.pooled_by_nearest_pct(x, y)
        bx, by = brute_pooled(x, y)
        assert np.allclose(px, bx) and np.allclose(py, by), n
    x = np.arange(10.0)
    px, py = rd.pooled_by_nearest_pct(x, x * 2)
    assert px[0] == 0.5 and py[0] == 1.0  # n = 10, half-width 1: first position averages points 0 and 1
    assert px[-1] == 8.5 and px[5] == 5.0

#-------------------------------------------------------------------------------

def test_table_renderer_equal_widths_and_split():
    rows = [(f"config{i}", [f"{i * 1.2345:.4f}", f"-{i}.5", "n/a"]) for i in range(30)]
    lines = sl.format_table("configuration", ["mean err (%)", "sd", "slope"], rows, separators={10})
    assert len({len(l) for l in lines}) == 1
    grouped = sl.format_table("configuration", ["a (%)", "b (%)"] * 2, [("x", ["1.00", "2.00", "3.00", "4.00"])],
                              groups=[("S = 2,500 and more text", 2), ("S = 10,000", 2)])
    assert len({len(l) for l in grouped}) == 1
    figs = sl.table_slides("t", ["sub"], lines, head_len=2, max_body=10)
    assert len(figs) > 1
    import matplotlib.pyplot as plt
    for f in figs:
        plt.close(f)

def test_subtitle_wrapper_never_exceeds_80():
    texts = ["word " * 60, "x" * 200, "Stage 3 (real database, real queries), SIFT (d = 128), S = 10,000, sample+train, "
             "seed 0, theta relative sd = 0.21%: read the mean error only", "short", ""]
    for t in texts:
        out = sl.wrap_subtitle([t, t + " tail"])
        assert out and all(len(l) <= 80 for l in out)
    assert len(sl.wrap_subtitle("a " * 100)) > 1

def test_glossary_entries_and_no_trailing_period():
    assert len(sl.GLOSSARY) == 17
    assert all(not e.rstrip().endswith(".") for e in sl.GLOSSARY)
    for width in (118, 80):
        lines = sl.glossary_lines(width)
        assert all(not l.rstrip().endswith(".") for l in lines)
        assert " ".join(l.strip() for l in lines).count(":") >= 17

def test_forbidden_words_absent_from_report_text():
    for name in ("rt_report_slides.py", "rt_report_data.py", "rt_report.py"):
        text = (HERE / name).read_text().lower()
        assert "window" not in text and "sliding" not in text, name

def test_title_slide_has_exact_subtitle_and_date_only():
    fig = sl.title_slide("sift")
    texts = [t.get_text() for t in fig.texts]
    assert texts[0] == "Multi-table LSH radius estimation: SIFT"
    assert texts[1] == "L tables per estimate, m-bit hashcodes, nu neighbors per table"
    assert len(texts) == 3 and re.match(r"[A-Z][a-z]+ \d{2}, \d{4}$", texts[2])
    import matplotlib.pyplot as plt
    plt.close(fig)

#-------------------------------------------------------------------------------
# Synthetic saved data

def synthetic_aggregates(theta_scale: float, n: int = 200, seeds=(0, 1, 2), rng_seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    theta = 0.4 + theta_scale * rng.normal(size=n)
    rows = []
    def row(config, L, m, S, pop, nu, seed, r_hat, scope="all_seeds", distinct=True):
        d = dict(config=config, L=L, m=m, S=S, pop=pop, nu=nu, seed=seed, distinct_scope=scope,
                 **rt_engine._stat_sums(theta, r_hat))
        d.update(n_distinct=n if distinct else np.nan, sum_distinct_total=7.0 * n if distinct else np.nan,
                 sum_distinct_train=3.0 * n if distinct else np.nan, sum_distinct_sample=4.0 * n if distinct else np.nan,
                 sum_train_share=0.4 * n if distinct else np.nan)
        return d
    rows.append(row("EXACT", 0, 0, -1, "none", 0, -1, theta, "none", False))
    for seed in seeds:
        for S in rt_config.S_GRID:
            rows.append(row("CONSTANT", 0, 0, S, "sample_only", 0, seed, np.full(n, 0.41), "none", False))
            for cname in rt_config.CONFIG_NAMES:
                L, m = (int(v[1:]) for v in cname.split("_"))
                for pop in rt_config.POP_KINDS:
                    for nu in (1, 8, 32):
                        r = 0.5 * theta + 0.2 + 0.001 * rng.normal(size=n)
                        rows.append(row(cname, L, m, S, pop, nu, seed, r))
    return pd.DataFrame(rows), theta

def test_mean_error_and_sd_from_aggregates_match_per_query():
    import rt_tables
    agg, theta = synthetic_aggregates(0.05)
    m = rt_tables.per_seed_metrics(agg)
    g = m[(m["config"] == "L4_m128") & (m["S"] == rt_config.REF_S) & (m["pop"] == "sample_train") & (m["nu"] == 8)]
    assert len(g) == 3
    # reconstruct r_hat from the sums is not possible; check the formulas on one synthetic row directly
    rng = np.random.default_rng(5)
    th = 0.4 + 0.05 * rng.normal(size=300)
    r = 0.5 * th + 0.2 + 0.01 * rng.normal(size=300)
    s = rt_engine._stat_sums(th, r)
    err = 100 * (r - th) / th
    row = pd.DataFrame([dict(config="c", L=1, m=1, S=1, pop="sample_train", nu=8, seed=0, distinct_scope="x", **s,
                             n_distinct=300, sum_distinct_total=300.0, sum_distinct_train=0.0, sum_distinct_sample=300.0,
                             sum_train_share=0.0)])
    mm = rt_tables.per_seed_metrics(row).iloc[0]
    assert np.isclose(mm["mean_error_pct"], err.mean()) and np.isclose(mm["sd_error_pct"], err.std())
    assert np.isclose(mm["slope"], np.polyfit(th, r, 1)[0])

def synthetic_data(theta_scale: float, stage: int) -> dict:
    import rt_tables
    agg, theta = synthetic_aggregates(theta_scale)
    return dict(dataset="sift", stage=stage, metrics=rt_tables.per_seed_metrics(agg))

def test_main_table_slide_alignment_decimals_and_slope_na():
    import matplotlib.pyplot as plt
    low = synthetic_data(0.0008, 1)   # relative sd about 0.2%: degenerate slope
    figs = sl.slide_table_main(low, 8, 4)
    body = [t.get_text() for f in figs for t in f.texts if "configuration" in t.get_text()]
    assert body and "n/a" in body[0]
    lines = body[0].split("\n")
    assert len({len(l) for l in lines}) == 1
    assert any(re.search(r"-?\d+\.\d{4}\b", l) for l in lines) and "constant radius" in body[0]
    wide = synthetic_data(0.06, 3)
    figs2 = sl.slide_table_main(wide, 8, 2)
    body2 = [t.get_text() for f in figs2 for t in f.texts if "configuration" in t.get_text()][0]
    assert "n/a" not in body2 and re.search(r"-?\d+\.\d{2}\b", body2)
    for f in figs + figs2:
        plt.close(f)

def test_train_points_table_matches_per_query_recomputation():
    rng = np.random.default_rng(2)
    frames = []
    for cname in rt_config.CONFIG_NAMES:
        for seed in range(3):
            n = 120
            theta = 0.2 + 0.05 * rng.random(n)
            r_st = theta * (1 + 0.02 * rng.normal(size=n))
            r_so = theta * (1 + 0.06 * rng.normal(size=n) - 0.03)
            ntr = rng.integers(0, 6, n)
            ntot = ntr + rng.integers(1, 6, n)
            frames.append(pd.DataFrame(dict(config=cname, seed=seed, nu=8, query_id=np.arange(n), n_distinct_train=ntr,
                                            n_distinct_total=ntot, err_st=100 * (r_st - theta) / theta,
                                            err_so=100 * (r_so - theta) / theta)))
    q = pd.concat(frames, ignore_index=True)
    out = {r["label"]: r for r in rd.table_train_points(q)}
    for cname in rt_config.CONFIG_NAMES:
        shares, smaller, change = [], [], []
        for seed in range(3):
            g = q[(q["config"] == cname) & (q["seed"] == seed)]
            shares.append((g["n_distinct_train"] / g["n_distinct_total"]).mean())
            smaller.append(np.mean(np.abs(g["err_st"].to_numpy()) < np.abs(g["err_so"].to_numpy())))
            change.append(np.mean(np.abs(g["err_so"].to_numpy()) - np.abs(g["err_st"].to_numpy())))
        assert np.isclose(out[cname]["train_share"], np.mean(shares))
        assert np.isclose(out[cname]["pct_smaller"], 100 * np.mean(smaller))
        assert np.isclose(out[cname]["mean_change_pp"], np.mean(change))

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
