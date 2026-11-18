#!/usr/bin/env python3
"""Untimed checks for cheaper ball fit / intruder count (doc 9ai), on saved 10% warm-up states:

  A. features from the 3 NEAREST balls (by center distance) instead of the 3 best-FITTING ones (and the
     intruder count over the 10 nearest instead of the 10 best-fitting): the model refit on the warm-up's fit
     split, the cutoff calibrated on its calibration split (as online_sim), hit rate / retention on the next
     --n requests at the target
  B. best fit searched among only the m nearest centers (m = 8 ... 1024): with the state's model and cutoff,
     decision agreement with the full scan and hit rate
  C. intruder count from RaBitQ distance estimates (1 bit per dimension after a random rotation, with
     RaBitQ's per-vector correction factors, one centroid = the sample mean) instead of exact distances:
     count agreement and decision agreement with the state's model and cutoff
Exact search everywhere else; one thread.

  PYTHONPATH=. .venv-core/bin/python simulations/approximate_hits/ballfit_experiments.py --workload tripclick-ip
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
import json
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np

RES = Path("simulations/approximate_hits/results")
N_BALLS, SHELL_BALLS, POOR_FIT, EPS, K = 3, 10, 10.0, 1e-12, 10


def knn_radius(xs, cv):
    import faiss
    d2, ii = faiss.knn(xs, cv, K)
    return np.sqrt(np.maximum(d2.max(axis=1), 0)) + EPS, ii


class Cell:
    def __init__(self, w, budget, n_test):
        from simulations.approximate_hits.prep_io import load_prep
        db, q, _, _ = load_prep(w.prep_dir)
        st = dict(np.load(w.state_dir / f"state_b{budget:g}.npz", allow_pickle=False))
        self.st = st
        self.cids = np.sort(np.asarray(st["cached_ids"], dtype=np.int64))
        self.cv = np.ascontiguousarray(np.asarray(db[self.cids], np.float32))
        self.cq = np.asarray(q[st["centers"]], np.float32)
        self.rad = st["radius"].astype(np.float64)
        self.sv = st["sv"].astype(np.float32)
        es, eo = st["entry_samp"], st["entry_samp_off"]
        self.esamp = [es[eo[j]:eo[j + 1]] for j in range(len(self.rad))]
        self.ans = np.load(w.answers_file)[f"answer_{w.target}"][:, :K]
        stream = np.load(w.prep_dir / "streams.npz")[w.stream]
        wu = int(st["warmup"])
        self.sets = {"fit": np.asarray(st["fit_q"], dtype=np.int64), "cal": np.asarray(st["cal_q"], dtype=np.int64),
                     "test": np.asarray(stream[wu:wu + n_test], dtype=np.int64)}
        self.cal_w = np.asarray(st["cal_w"], dtype=np.float64)
        self.q = q
        self.data = {}
        for name, qi in self.sets.items():
            xs = np.ascontiguousarray(np.asarray(q[qi], np.float32))
            nr, nn = knn_radius(xs, self.cv)
            de = np.sqrt(np.maximum(((xs[:, None, :] - self.cq[None, :, :]) ** 2).sum(axis=2), 0)) \
                if len(self.cq) * len(xs) < 2e8 else self._de_chunked(xs)
            y = np.array([np.isin(self.ans[i], self.cids[n]).mean() for i, n in zip(qi, nn)])
            self.data[name] = {"xs": xs, "nr": nr, "de": de, "y": y}

    def _de_chunked(self, xs):
        out = np.empty((len(xs), len(self.cq)))
        c2 = (self.cq.astype(np.float64) ** 2).sum(axis=1)
        for s in range(0, len(xs), 512):
            x = xs[s:s + 512].astype(np.float64)
            out[s:s + 512] = np.sqrt(np.maximum((x * x).sum(axis=1)[:, None] - 2 * x @ self.cq.T.astype(np.float64) + c2, 0))
        return out

    def features(self, name, order="fit", m=None, counter=None):
        """7 features per request: 3 balls chosen by best fit (among the m nearest centers if m) or by
        nearest center; intruders over the 10 chosen likewise; counter(x, sample positions, nr) -> count."""
        D = self.data[name]
        X = np.full((len(D["nr"]), 2 * N_BALLS + 1), POOR_FIT)
        for i in range(len(D["nr"])):
            nr, de = D["nr"][i], D["de"][i]
            fit = (nr + de) / self.rad
            if order == "near":
                cand = np.argsort(de, kind="stable")
                sel = cand[:max(N_BALLS, SHELL_BALLS)]
            else:
                pool = np.argpartition(de, m - 1)[:m] if m and m < len(de) else np.arange(len(de))
                sel = pool[np.argsort(fit[pool], kind="stable")[:max(N_BALLS, SHELL_BALLS)]]
            for j, e in enumerate(sel[:N_BALLS]):
                X[i, 2 * j], X[i, 2 * j + 1] = de[e] / self.rad[e], nr / self.rad[e]
            pos = np.unique(np.concatenate([self.esamp[e] for e in sel[:SHELL_BALLS]])) if len(sel) else np.empty(0, int)
            if not len(pos):
                cnt = 0
            elif counter is None:
                cnt = int((((self.sv[pos] - D["xs"][i]) ** 2).sum(axis=1) < nr * nr).sum())
            else:
                cnt = counter(D["xs"][i], pos, nr)
            X[i, -1] = np.log1p(cnt)
        return X

    def state_scores(self, X):
        st = self.st
        return ((X - st["mu"]) / st["sd"]) @ st["coef"] + float(st["intercept"])


class RaBitQ:
    """1-bit RaBitQ (Gao & Long, SIGMOD 2024) for the shell sample: o' = P^T (o - c) / ||o - c||, code
    x = sign(o') / sqrt(D); <o', q'> estimated as <x, P^T q'> / <x, o'> (unbiased); distance from it."""

    def __init__(self, sv, seed=0):
        rng = np.random.default_rng(seed)
        D = sv.shape[1]
        self.c = sv.mean(axis=0)
        self.P, _ = np.linalg.qr(rng.normal(size=(D, D)))
        r = sv.astype(np.float64) - self.c
        self.norm = np.linalg.norm(r, axis=1)
        u = (r / np.maximum(self.norm, EPS)[:, None]) @ self.P
        self.code = np.where(u >= 0, 1.0, -1.0) / np.sqrt(D)
        self.xo = (self.code * u).sum(axis=1)                 # <x, o'>
        self.code = self.code.astype(np.float32)

    def count(self, x, pos, nr):
        qr = x.astype(np.float64) - self.c
        qn = np.linalg.norm(qr)
        qp = (qr / max(qn, EPS)) @ self.P
        ip = (self.code[pos] @ qp) / np.maximum(self.xo[pos], EPS)
        d2 = self.norm[pos] ** 2 + qn * qn - 2 * self.norm[pos] * qn * ip
        return int((d2 < nr * nr).sum())


def calibrated_hits(cell, Xf, Xc, Xt, T=0.9, delta=0.05):
    """online_sim's warm-up rule: model fit on the fit split, start from the fit split, cutoff from the
    calibration split (weighted), applied to the test requests."""
    from simulations.approximate_hits.eval_generalize import fit_logistic
    from simulations.approximate_hits.threshold_selection import select_weighted, threshold_grid, start_from_fit
    yf, yc, yt = cell.data["fit"]["y"], cell.data["cal"]["y"], cell.data["test"]["y"]
    model = fit_logistic(Xf, yf)
    start = start_from_fit(model(Xf), yf, T, delta)
    sc = model(Xc)
    lam = select_weighted(sc, yc, cell.cal_w[:len(yc)], threshold_grid(sc), T, delta, start=start)
    if lam is None:
        return {"lambda": None, "hit_rate": 0.0, "retention": None}
    h = model(Xt) >= lam
    return {"lambda": float(lam), "hit_rate": float(h.mean()), "retention": float(yt[h].mean()) if h.any() else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--budget", type=float, default=0.1)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, help="D: only the refit / recalibrated comparison of RaBitQ counts")
    a = ap.parse_args()
    from simulations.approximate_hits.workloads import get
    w = get(a.workload)
    cell = Cell(w, a.budget, a.n)
    res = {"workload": w.name, "budget": a.budget, "n_entries": int(len(cell.rad)), "n_cached": int(len(cell.cids)),
           "n_samples": int(len(cell.sv)), "dim": int(cell.cv.shape[1])}
    lam = float(cell.st["lam"])
    if a.only == "D":
        # D. RaBitQ intruder counts with the model refit and the cutoff recalibrated on them (as A): does the
        # model tolerate the count error at the target?
        rq = RaBitQ(cell.sv)
        res["D"] = {}
        for name, counter in (("4-bit exact", None), ("RaBitQ 1-bit", rq.count)):
            Xs = [cell.features(s_, counter=counter) for s_ in ("fit", "cal", "test")]
            res["D"][name] = calibrated_hits(cell, *Xs)
            print(json.dumps({"D": name, **res["D"][name]}), flush=True)
        out = RES / "ballfit_experiments" / f"{w.name}_b{a.budget:g}_D.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1))
        return
    # A. nearest vs best-fitting balls, each with its own refit model and calibrated cutoff
    res["A"] = {}
    for order in ("fit", "near"):
        Xs = [cell.features(s, order=order) for s in ("fit", "cal", "test")]
        res["A"][order] = calibrated_hits(cell, *Xs)
        print(json.dumps({"A": order, **res["A"][order]}), flush=True)
    # B. best fit among the m nearest centers (state model and cutoff)
    X0 = cell.features("test")
    h0 = cell.state_scores(X0) >= lam
    yt = cell.data["test"]["y"]
    res["B"] = {"full_scan": {"hit_rate": float(h0.mean()), "retention": float(yt[h0].mean()) if h0.any() else None}}
    for m in (1024, 512, 256, 128, 64, 32, 16, 8):
        if m >= len(cell.rad):
            continue
        h = cell.state_scores(cell.features("test", m=m)) >= lam
        res["B"][m] = {"agree": float((h == h0).mean()), "hit_rate": float(h.mean()),
                       "retention": float(yt[h].mean()) if h.any() else None}
        print(json.dumps({"B_m": m, **res["B"][m]}), flush=True)
    # C. RaBitQ intruder count (state model and cutoff)
    rq = RaBitQ(cell.sv)
    Xr = cell.features("test", counter=rq.count)
    hr = cell.state_scores(Xr) >= lam
    cnt0, cntr = np.expm1(X0[:, -1]), np.expm1(Xr[:, -1])
    res["C"] = {"count_equal": float(np.mean(np.rint(cnt0) == np.rint(cntr))),
                "count_mean_abs_diff": float(np.mean(np.abs(cnt0 - cntr))), "count_mean": float(cnt0.mean()),
                "decision_agree": float((hr == h0).mean()), "hit_rate": float(hr.mean()),
                "retention": float(yt[hr].mean()) if hr.any() else None}
    print(json.dumps({"C": res["C"]}), flush=True)
    out = Path(a.out) if a.out else RES / "ballfit_experiments" / f"{w.name}_b{a.budget:g}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
