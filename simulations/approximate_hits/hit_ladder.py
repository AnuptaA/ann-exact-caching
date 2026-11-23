#!/usr/bin/env python3
"""Hit-rate ladder (doc 9al) on the feature tables of hit_table.py: for every variant of the hit decision,
the hit rate at the retention target T on held-out requests, so that what costs little hit rate can be
cut for latency.

Requests after the warm-up (with repeats, as the stream has them) are split at random (per draw) into a
training pool (30%), a calibration pool (30%) and a test set (40%). Per variant and draw:
  oracle     the best hit rate any decision could get on the test set given the served answers (serve the
             highest-retention requests while their mean retention stays >= T)
  bestcut    the model (logistic, soft labels, fit on n_fit training requests) with the best cutoff for
             the test set: the model's ranking quality
  cal        the model with lambda calibrated on n_cal calibration requests (online_sim's rule: weighted
             normal bound, fixed order, grid from the calibration scores, start from the fit split):
             test hit rate, retention, and how often retention < T - 0.01 across draws
Every request is labeled here (as --oracle-cal); in a deployment only misses and audited hits are.
Also "deployed": the warm-up's own fit and calibration splits, tested on all post-warm-up requests.

  PYTHONPATH=. .venv/bin/python simulations/approximate_hits/hit_ladder.py results/hit_tables/<w>/<file>.npz
  ... --summary   (all tables -> results/hit_tables/summary.json / summary.md)
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np

from simulations.approximate_hits.eval_generalize import fit_logistic, optimal_rate
from simulations.approximate_hits.threshold_selection import select_weighted, threshold_grid, start_from_fit

RES = Path("simulations/approximate_hits/results")
N_FIT, N_CAL, DRAWS = 4000, 4000, 5
FIT_SWEEP = (250, 500, 1000, 2000, 4000, 8000, 16000)
CAL_SWEEP = (250, 500, 1000, 2000, 4000, 8000, 16000)


class Table:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.z = {k: z[k] for k in z.files}
        aug5 = Path(path).with_suffix(".aug5.npz")
        if aug5.exists():                # hit_table_aug5.py: full-shell counts from DiskANN's PQ codes / float (doc 9au)
            a = np.load(aug5)
            self.z.update({k: a[k] for k in a.files})
        aug4 = Path(path).with_suffix(".aug4.npz")
        if aug4.exists():
            a = np.load(aug4)
            self.z.update({k: a[k] for k in a.files})
        aug3 = Path(path).with_suffix(".aug3.npz")
        if aug3.exists():                # hit_table_aug3.py: serving from the union of the m best-fitting balls
            a = np.load(aug3)
            self.z.update({k: a[k] for k in a.files})
        aug2 = Path(path).with_suffix(".aug2.npz")
        if aug2.exists():                # hit_table_aug2.py: exact intruder count (not deployable)
            a = np.load(aug2)
            self.z.update({k: a[k] for k in a.files})
        aug = Path(path).with_suffix(".aug.npz")
        if aug.exists():                 # hit_table_aug.py: 8-bit / robust-range counts over the whole sample
            a = np.load(aug)
            self.z.update({k: a[k] for k in a.files})
        self.meta = json.loads(str(self.z["meta"]))
        self.rad = self.z["rad"]
        self.ip0 = self.meta["p_mult"].index(1.0)

    def balls(self, nb=3, order="fit", g=None):
        """[d / radius, nr / radius] for the nb best-fitting (or nearest) entries; g: graph nr / fit order."""
        z = self.z
        if g is None:
            idx, de, nr = z[f"{order}_idx"], z[f"{order}_de"], z["nr"]
        else:
            nr = z[f"{g}_nr"]
            idx, de = (z[f"{g}_fit_idx"], z[f"{g}_fit_de"]) if order == "fit" else (z["near_idx"], z["near_de"])
        cols = []
        for j in range(nb):
            r = self.rad[idx[:, j]] if len(self.rad) else np.ones(len(nr))
            ok = j < len(self.rad)
            cols += [np.where(ok, de[:, j] / r, 10.0), np.where(ok, nr / r, 10.0)]
        return np.column_stack(cols)

    def balls_near_fit(self, nb, m):
        """Best fit (nr + d) / radius among the m nearest centers (from the 10 stored), nb best of them."""
        z = self.z
        idx, de, nr = z["near_idx"][:, :m], z["near_de"][:, :m].astype(np.float64), z["nr"]
        rad = self.rad[idx]
        fit = (nr[:, None] + de) / rad
        o = np.argsort(fit, axis=1, kind="stable")[:, :nb]
        cols = []
        for j in range(nb):
            if j >= o.shape[1]:
                cols += [np.full(len(nr), 10.0)] * 2
                continue
            cols += [np.take_along_axis(de / rad, o[:, j:j + 1], axis=1)[:, 0], nr / np.take_along_axis(rad, o[:, j:j + 1], axis=1)[:, 0]]
        return np.column_stack(cols)

    def count(self, name, pm=1.0):
        z = self.z
        v = z[name]
        if v.ndim == 2:
            v = v[:, self.meta["p_mult"].index(pm)]
        return np.log1p(v.astype(np.float64))[:, None]

    def variants(self):
        """name -> (X over distinct requests, label y, group)."""
        z, V = self.z, {}
        B3 = self.balls(3)
        V["ref: 3 balls + count over all samples (4-bit)"] = (np.hstack([B3, self.count("all_4")]), z["y"], "ref")
        # model / features
        V["features: count float (all samples)"] = (np.hstack([B3, self.count("all_f")]), z["y"], "features")
        for key, lab in (("all_8", "8-bit"), ("all_4r", "4-bit, 1-99% range"), ("all_8r", "8-bit, 1-99% range")):
            if key in z:
                V[f"features: count {lab} (all samples)"] = (np.hstack([B3, self.count(key)]), z["y"], "features")
        V["features: 1 ball + count"] = (np.hstack([self.balls(1), self.count("all_4")]), z["y"], "features")
        V["features: 2 balls + count"] = (np.hstack([self.balls(2), self.count("all_4")]), z["y"], "features")
        V["features: 3 balls, no count"] = (B3, z["y"], "features")
        V["features: 1 ball, no count"] = (self.balls(1), z["y"], "features")
        V["features: count only"] = (self.count("all_4"), z["y"], "features")
        # shell scope / size / codes (fit order)
        for B in (3, 10):
            for pm in self.meta["p_mult"]:
                V[f"shell: {B} best-fit balls, p x{pm:g}, 4-bit"] = (np.hstack([B3, self.count(f"cnt_fit{B}_4", pm)]), z["y"], "shell")
            V[f"shell: {B} best-fit balls, p x1, float"] = (np.hstack([B3, self.count(f"cnt_fit{B}_f", 1.0)]), z["y"], "shell")
            V[f"shell: {B} best-fit balls, p x1, RaBitQ"] = (np.hstack([B3, self.count(f"rq_fit{B}")]), z["y"], "shell")
        # ball finder: nearest centers (features and shells from the nearest entries)
        N3 = self.balls(3, "near")
        V["balls: 3 nearest, count over all"] = (np.hstack([N3, self.count("all_4")]), z["y"], "balls")
        for B in (3, 10):
            V[f"balls: 3 nearest, shell {B} nearest, p x1, 4-bit"] = (np.hstack([N3, self.count(f"cnt_near{B}_4", 1.0)]), z["y"], "balls")
            V[f"balls: 3 nearest, shell {B} nearest, p x0.25, 4-bit"] = (np.hstack([N3, self.count(f"cnt_near{B}_4", 0.25)]), z["y"], "balls")
        V["balls: 1 nearest + count over all"] = (np.hstack([self.balls(1, "near"), self.count("all_4")]), z["y"], "balls")
        V["balls: 2 nearest + count over all"] = (np.hstack([self.balls(2, "near"), self.count("all_4")]), z["y"], "balls")
        for mm in (3, 10):
            V[f"balls: best fit among {mm} nearest, 1 ball + count over all"] = (np.hstack([self.balls_near_fit(1, mm), self.count("all_4")]), z["y"], "balls")
            V[f"balls: best fit among {mm} nearest, 3 balls + count over all"] = (np.hstack([self.balls_near_fit(3, mm), self.count("all_4")]), z["y"], "balls")
        V["balls: 1 nearest, shell 3 nearest, p x1, 4-bit"] = (np.hstack([self.balls(1, "near"), self.count("cnt_near3_4", 1.0)]), z["y"], "balls")
        # graph search (stage-1 radius / fit order; stage-2 served answer as the label)
        for g in self.meta.get("graph", []):
            t = f"g{g[0]}_{g[1]}"
            Xg = np.hstack([self.balls(3, "fit", t), np.log1p(z[f"{t}_all_4"].astype(float))[:, None]])
            V[f"graph ef {g[0]}/{g[1]}: features only"] = (Xg, z["y"], "graph")
            V[f"graph ef {g[0]}/{g[1]}: features + served answer"] = (Xg, z[f"{t}_y"], "graph")
            Xg10 = np.hstack([self.balls(3, "fit", t), np.log1p(z[f"{t}_fit10_4"].astype(float))[:, None]])
            V[f"graph ef {g[0]}/{g[1]}: + shell 10 best-fit (timing v6)"] = (Xg10, z[f"{t}_y"], "graph")
            Xn = np.hstack([self.balls(3, "near", t), np.log1p(z[f"{t}_near10_4"].astype(float))[:, None]])
            V[f"graph ef {g[0]}/{g[1]}: + 3 nearest, shell 10 nearest (timing near)"] = (Xn, z[f"{t}_y"], "graph")
        # headroom probes (not deployed): richer features; the exact intruder count
        rich = [self.balls(min(10, z["fit_idx"].shape[1]))] + [self.count(f"cnt_fit{B}_f", pm) for B in (3, 10) for pm in self.meta["p_mult"]]
        rich += [self.count("all_f"), self.count("all_4"), self.count("rq_fit10")]
        V["probe: rich features (10 balls, counts at every p, float, RaBitQ)"] = (np.hstack(rich), z["y"], "probe")
        if "exact_cnt" in z:
            V["probe: 3 balls + exact intruder count"] = (np.hstack([B3, np.log1p(z["exact_cnt"].astype(float))[:, None]]), z["y"], "probe")
            V["probe: 1 ball + exact intruder count"] = (np.hstack([self.balls(1), np.log1p(z["exact_cnt"].astype(float))[:, None]]), z["y"], "probe")
        if "pq_shell10" in z:            # doc 9au: the backend's compressed codes instead of our shell sample
            V["pq: 10 best-fit balls, whole shell, DiskANN PQ codes"] = (np.hstack([B3, np.log1p(z["pq_shell10"].astype(float))[:, None]]), z["y"], "pq")
            V["pq: 10 best-fit balls, whole shell, float (ceiling)"] = (np.hstack([B3, np.log1p(z["f_shell10"].astype(float))[:, None]]), z["y"], "pq")
        # serving from one entry instead of the whole cache (features as ref)
        V["serve: best-fit entry only"] = (V[next(iter(V))][0], z["y_single_fit"], "serve")
        for mm in (1, 3, 10, 30, 100):
            if f"y_union{mm}" in z:
                V[f"serve: union of {mm} best-fit balls (exact, scan {int(z[f'scan_union{mm}'].mean())} vectors)"] = (V[next(iter(V))][0], z[f"y_union{mm}"], "serve")
        V["serve: nearest entry only"] = (np.hstack([N3, self.count("all_4")]), z["y_single_near"], "serve")
        return V

    def cost(self):
        """Mean shell vectors examined per request (cost proxy) per scope / sampling rate."""
        z, out = self.z, {}
        for o in ("fit", "near"):
            for B in (3, 10):
                v = z[f"scan_{o}{B}"]
                out[f"{o}{B}"] = dict(zip([f"x{m:g}" for m in self.meta["p_mult"]], v.mean(axis=0).round(1).tolist()))
        out["all"] = self.meta["samples_p0"]
        return out


def best_cut(s, y, T):
    o = np.argsort(-s, kind="stable")
    ys = y[o]
    ok = np.cumsum(ys) / np.arange(1, len(ys) + 1) >= T - 1e-12
    return float((np.flatnonzero(ok).max() + 1) / len(ys)) if ok.any() else 0.0


def calibrated(model, Xf, yf, Xc, yc, Xt, yt, T, delta):
    sf, sc = model(Xf), model(Xc)
    lam = select_weighted(sc, yc, np.ones(len(yc)), threshold_grid(sc), T, delta, start=start_from_fit(sf, yf, T, delta))
    if lam is None:
        return 0.0, None
    h = model(Xt) >= lam
    return float(h.mean()), (float(yt[h].mean()) if h.any() else None)


def evaluate(tab, X, y, rows, T, delta, n_fit=N_FIT, n_cal=N_CAL, draws=DRAWS, extra=False):
    """rows: post-warm-up requests as row indices of X (with repeats)."""
    res = {"oracle": [], "bestcut": [], "cal_hit": [], "cal_ret": [], "below": 0}
    for d in range(draws):
        rng = np.random.default_rng(1000 + d)
        perm = rng.permutation(len(rows))
        a, b = int(0.3 * len(rows)), int(0.6 * len(rows))
        tr, ca, te = rows[perm[:a]], rows[perm[a:b]], rows[perm[b:]]
        tr, ca = tr[:n_fit], ca[:n_cal]
        yt = y[te]
        res["oracle"].append(optimal_rate(yt, T))
        if np.ptp(y[tr]) == 0 or X.shape[1] == 0:
            res["bestcut"].append(0.0); res["cal_hit"].append(0.0); res["cal_ret"].append(None)
            continue
        model = fit_logistic(X[tr], y[tr])
        res["bestcut"].append(best_cut(model(X[te]), yt, T))
        h, r = calibrated(model, X[tr], y[tr], X[ca], y[ca], X[te], yt, T, delta)
        res["cal_hit"].append(h); res["cal_ret"].append(r)
        res["below"] += int(r is not None and r < T - 0.01)
    rets = [r for r in res["cal_ret"] if r is not None]
    return {"oracle": float(np.mean(res["oracle"])), "bestcut": float(np.mean(res["bestcut"])),
            "cal": float(np.mean(res["cal_hit"])), "cal_ret": float(np.mean(rets)) if rets else None,
            "cal_below": res["below"], "draws": draws}


def run_table(path, quick=False):
    tab = Table(path)
    z, m = tab.z, tab.meta
    T, delta = m["T"], m["delta"]
    rows = z["post_q"]
    V = tab.variants()
    out = {"meta": m, "variants": {}, "cost": tab.cost()}
    draws = 2 if quick else DRAWS
    for name, (X, y, grp) in V.items():
        r = evaluate(tab, X, y, rows, T, delta, draws=draws)
        r["group"] = grp
        out["variants"][name] = r
    Xr, yr, _ = V[next(iter(V))]
    out["fit_sweep"] = {n: evaluate(tab, Xr, yr, rows, T, delta, n_fit=n, draws=draws) for n in FIT_SWEEP}
    out["cal_sweep"] = {n: evaluate(tab, Xr, yr, rows, T, delta, n_cal=n, draws=draws) for n in CAL_SWEEP}
    # deployed: the warm-up's own fit / calibration splits, all post-warm-up requests as test
    fq, cq = z["fit_q"], z["cal_q"]
    model = fit_logistic(Xr[fq], yr[fq])
    h, r = calibrated(model, Xr[fq], yr[fq], Xr[cq], yr[cq], Xr[rows], yr[rows], T, delta)
    out["deployed"] = {"cal": h, "cal_ret": r, "oracle": optimal_rate(yr[rows], T)}
    # oracle of single-entry serving: the best of the 10 best-fitting entries
    out["oracle_single_best10"] = optimal_rate(z["y_single_best"][rows], T)
    return out


def summary():
    """All ladders -> summary.json and a markdown table: per variant, calibrated hit rate relative to the
    reference variant of the same table (mean / min over cells), and the cross-table comparisons
    (backend, cache variants) for the reference variant."""
    base = RES / "hit_tables"
    L = {}
    for f in sorted(base.glob("*/*.ladder.json")):
        d = json.loads(f.read_text())
        m = d["meta"]
        # the cell's tuned search lists (tune_core) -> one row "graph tuned"
        tf = RES / "tuning_core" / m["workload"] / f"b{m['budget']:g}.json"
        if tf.exists():
            t = json.loads(tf.read_text())
            pre = f"graph ef {t['stage1']['ef']}/{t['stage2']['ef']}:"
            d["variants"].update({n.replace(pre, "graph tuned:"): v for n, v in list(d["variants"].items()) if n.startswith(pre)})
        fixed = {f"graph ef {a}/{b}:" for a, b in ((16, 64), (64, 256), (256, 1024))}
        d["variants"] = {n: v for n, v in d["variants"].items()
                         if not n.startswith("graph ef") or any(n.startswith(p) for p in fixed)}
        L[(m["workload"], m["budget"], m["variant"], m["backend"])] = d
    return _summary(base, L)


KEY = ["features: 1 ball + count", "features: 2 balls + count", "features: count float (all samples)",
       "shell: 10 best-fit balls, p x1, 4-bit", "shell: 10 best-fit balls, p x0.25, 4-bit",
       "shell: 10 best-fit balls, p x1, RaBitQ", "balls: 3 nearest, shell 10 nearest, p x1, 4-bit",
       "graph tuned: features + served answer", "graph tuned: + shell 10 best-fit (timing v6)",
       "serve: best-fit entry only"]


def _summary(base, L):
    lines = ["# Hit-rate ladder (doc 9al)", "",
             "Calibrated hit rate at T relative to the reference variant of the same cache (3 best-fit balls + "
             "count over the whole 4-bit sample, exact search), default cache, exact backend. mean / min over "
             "cells; 'below' = draws with test retention < T - 0.01.", ""]
    ref_name = None
    rows = {}
    for key, d in L.items():
        if key[2] != "default" or key[3] != "exact":
            continue
        names = list(d["variants"])
        ref_name = names[0]
        ref = d["variants"][ref_name]
        for n in names:
            v = d["variants"][n]
            rr = rows.setdefault(n, {"rel": [], "below": 0, "draws": 0, "abs": [], "oracle_rel": []})
            rr["rel"].append(v["cal"] / ref["cal"] if ref["cal"] > 0 else np.nan)
            rr["abs"].append(v["cal"])
            rr["oracle_rel"].append(v["cal"] / v["oracle"] if v["oracle"] > 0 else np.nan)
            rr["below"] += v["cal_below"]
            rr["draws"] += v["draws"]
    lines += ["| variant | rel. to ref (mean) | min | share of its oracle (mean) | below / draws |", "|---|---|---|---|---|"]
    for n, rr in rows.items():
        rel = np.array(rr["rel"], float)
        orc = np.array(rr["oracle_rel"], float)
        lines.append(f"| {n} | {np.nanmean(rel):.3f} | {np.nanmin(rel):.3f} | {np.nanmean(orc):.3f} | "
                     f"{rr['below']} / {rr['draws']} |")
    # cross-table: the same reference variant on other caches / backends
    lines += ["", "Reference variant, calibrated hit rate (absolute %), per cell: default exact / diskann backend / "
              "maxk10 / f95 / f100; oracle of the default cache (whole cache / best single entry)", "",
              "| workload | budget | default | diskann | maxk10 | f95 | f100 | oracle | oracle single |", "|---|---|---|---|---|---|---|---|---|"]
    cells = sorted({(k[0], k[1]) for k in L})
    for w, b in cells:
        def g(var, be):
            d = L.get((w, b, var, be))
            return f"{100 * d['variants'][next(iter(d['variants']))]['cal']:.1f}" if d else "-"
        dd = L.get((w, b, "default", "exact"))
        orc = f"{100 * dd['variants'][next(iter(dd['variants']))]['oracle']:.1f}" if dd else "-"
        ors = f"{100 * dd['oracle_single_best10']:.1f}" if dd else "-"
        lines.append(f"| {w} | {b:g} | {g('default', 'exact')} | {g('default', 'diskann')} | {g('maxk10', 'exact')} | "
                     f"{g('f95', 'exact')} | {g('f100', 'exact')} | {orc} | {ors} |")
    # training / calibration sizes (reference variant), relative to the default sizes
    lines += ["", f"Training / calibration size (reference variant; relative to n_fit = n_cal = {N_FIT}), mean / min over cells:", "",
              "| n | training: rel. | min | calibration: rel. | min | calibration: below / draws |", "|---|---|---|---|---|---|"]
    de = [d for k, d in L.items() if k[2] == "default" and k[3] == "exact"]
    for n in FIT_SWEEP:
        fr = [d["fit_sweep"][str(n)]["cal"] / d["fit_sweep"][str(N_FIT)]["cal"] for d in de if d["fit_sweep"][str(N_FIT)]["cal"] > 0]
        cr = [d["cal_sweep"][str(n)]["cal"] / d["cal_sweep"][str(N_CAL)]["cal"] for d in de if d["cal_sweep"][str(N_CAL)]["cal"] > 0]
        cb = sum(d["cal_sweep"][str(n)]["cal_below"] for d in de)
        cd = sum(d["cal_sweep"][str(n)]["draws"] for d in de)
        lines.append(f"| {n} | {np.mean(fr):.3f} | {np.min(fr):.3f} | {np.mean(cr):.3f} | {np.min(cr):.3f} | {cb} / {cd} |")
    dep = [d["deployed"]["cal"] / d["variants"][next(iter(d["variants"]))]["cal"] for d in de
           if d["variants"][next(iter(d["variants"]))]["cal"] > 0]
    lines += ["", f"As deployed (warm-up fit / calibration splits, ~4,000 each), relative to the reference: mean "
              f"{np.mean(dep):.3f}, min {np.min(dep):.3f}; deployed retention below T - 0.01 in "
              f"{sum(1 for d in de if (d['deployed']['cal_ret'] or 1) < d['meta']['T'] - 0.01)} of {len(de)} cells."]
    # per workload: the main cuts, relative to the reference (mean over the workload's sizes)
    wl = sorted({k[0] for k in L})
    lines += ["", "Main cuts per workload (calibrated hit rate relative to the reference, mean over sizes):", "",
              "| variant | " + " | ".join(wl) + " |", "|---|" + "---|" * len(wl)]
    for n in ["ref oracle share"] + KEY:
        cells = []
        for w in wl:
            vals = []
            for k, d in L.items():
                if k[0] != w or k[2] != "default" or k[3] != "exact":
                    continue
                V = d["variants"]
                r = V[next(iter(V))]
                if n == "ref oracle share":
                    vals.append(r["cal"] / r["oracle"] if r["oracle"] > 0 else np.nan)
                elif n in V and r["cal"] > 0:
                    vals.append(V[n]["cal"] / r["cal"])
            cells.append(f"{np.nanmean(vals):.2f}" if vals else "-")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    (base / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tables", nargs="*")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    for t in a.tables:
        out = run_table(t, quick=a.quick)
        p = Path(t).with_suffix(".ladder.json")
        p.write_text(json.dumps(out, default=float))
        print(f"wrote {p}")
        ref = next(iter(out["variants"]))
        for n, v in out["variants"].items():
            print(f"{n:72s} oracle {100 * v['oracle']:5.1f} bestcut {100 * v['bestcut']:5.1f} "
                  f"cal {100 * v['cal']:5.1f} ret {v['cal_ret'] or 0:.3f} below {v['cal_below']}/{v['draws']}")
        print("deployed", out["deployed"], "oracle single best-of-10", out["oracle_single_best10"])
        print("fit sweep", {n: round(100 * v["cal"], 1) for n, v in out["fit_sweep"].items()})
        print("cal sweep", {n: round(100 * v["cal"], 1) for n, v in out["cal_sweep"].items()})
    if a.summary:
        summary()


if __name__ == "__main__":
    main()
