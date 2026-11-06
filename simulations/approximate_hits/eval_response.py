#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Detection + response study (context doc 5w): does tolerating false alarms
pay off when the response to an alarm is cheap and verifies itself?

Replays sampled streams as in eval_detectors.py (P pre-shift queries from
group A, then H post-shift queries from (1 - eps) A + eps B), now with the
serving loop's calibration dynamics:
  - lambda recalibrated every batch (select_weighted) on labeled pairs of
    the last --cal-span queries since the last reset: misses (weight 1)
    and audited hits (weight 1 / audit rate); audits adaptive (aim at A
    audits per batch, rate >= 0.1);
  - detection (after every batch): score test (KS, last 2000 scores vs the
    earlier ones since the last reset) and retention test (normal bound,
    audits of the last 1k / 2k / 5k / 10k / 20k queries); level per check:
    strict = delta / (2 j (j + 1)) per test (false alarms <= delta per run),
    or loose = a fixed per-check level (false alarms accepted);
  - response to an alarm (recalibration): reset = only labels after the
    alarm (nothing served until they arrive); grace g = keep serving at the
    pre-alarm lambda for g batches while fresh labels accumulate, then
    fresh-only; boost = audit target raised to 100 per batch for 5 batches;
  - shadow rebuild (optional): 5000 and 15000 queries after an alarm, a
    candidate cache built from that window (precomputed per (mix, window)
    by --prepare: misses + weighted audited hits of a window sampled from
    the mix, 20% held out for its calibration, greedy as online_sim) is
    compared with the current cache by oracle hits on half of the window;
    installed iff >= 10% better; then scores / retentions come from the
    candidate's arrays and calibration restarts on its held-out part.
Build queries are disjoint from the replayed queries. Metrics per (config,
eps): post-shift hit rate and served retention, share of 5000-query
windows below T and the worst one, pre-shift hit rate (cost of false
alarms), alarms, rebuilds, audits per 1000 queries.
"""

import sys
import json
import argparse
import logging
import itertools
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

RES = Path("simulations/approximate_hits/results")
T, DELTA = 0.9, 0.05
EPS_GRID = (0.0, 0.1, 0.3, 0.5, 0.7, 1.0)
WINDOWS = (5000, 15000)
SCAN = (1000, 2000, 5000, 10000, 20000)
B = 1000
CAL_SPAN = 20000
_G: dict = {}

#-------------------------------------------------------------------------------
# prepare


def prepare(prep, out, n_warm=20000, build_share=0.3):
    from simulations.approximate_hits import online_sim as osim
    a = argparse.Namespace(k=10, target=T, delta=0.05, budget=0.1 if "esci" in prep else 0.01,
                           f=0.85, R=500, bits=4, warmup=n_warm, seed=0, cal_span=CAL_SPAN,
                           n_db=1000000, cand_share=0.2, c_max=3000, s_max=25000, gain=0.1)
    D, streams = osim.load(RES / "online" / prep, a)
    labels = np.load(RES / "online" / prep / "clusters.npy")
    group = streams["group"]
    trace = "trace" in streams.files
    requests = streams["trace"] if trace else np.random.default_rng(0).permutation(len(labels))
    qg = group[labels[requests]]
    pos_a = np.flatnonzero(qg == 0)
    warm = requests[pos_a[:n_warm]]
    sim = osim.Sim(D, warm, a)
    sim.warmup()
    lam0 = sim.lam
    rng = np.random.default_rng(5)
    # distinct queries after warm-up, split into build / eval (disjoint)
    qa = np.unique(requests[pos_a[n_warm:]])
    qb = np.unique(requests[qg == 1])
    if not trace:
        qa = np.setdiff1d(qa, warm)
    split = lambda q: (lambda m: (q[m], q[~m]))(rng.random(len(q)) < build_share)
    build_a, eval_a = split(qa)
    build_b, eval_b = split(qb)
    # request weights (traces: frequency of each distinct query)
    freq = np.bincount(requests, minlength=len(D["q"]))
    ev = np.r_[eval_a, eval_b]

    def arrays(cache, model, idx):
        s, y = np.empty(len(idx)), np.empty(len(idx))
        for i in range(0, len(idx), 8192):
            X, _ = cache.features(idx[i:i + 8192])
            s[i:i + 8192], y[i:i + 8192] = model(X), cache.retention(idx[i:i + 8192])
        return s, y
    s0, y0 = arrays(sim.cache, sim.model, ev)
    pos = {int(q): i for i, q in enumerate(ev)}
    res = {"lam0": lam0, "eval_a": eval_a, "eval_b": eval_b, "freq": freq[ev], "s0": s0,
           "y0": y0, "cal0_s": sim.cal[0][1], "cal0_y": sim.cal[0][2]}
    # score / retention of build queries under cache 0 (for window labels)
    bq = np.r_[build_a, build_b]
    bs0, by0 = arrays(sim.cache, sim.model, bq)
    bpos = {int(q): i for i, q in enumerate(bq)}
    for eps, W in itertools.product(EPS_GRID, WINDOWS):
        n_b = int(round(eps * W))
        pa = freq[build_a] / freq[build_a].sum()
        pb = freq[build_b] / freq[build_b].sum()
        win = np.r_[rng.choice(build_a, W - n_b, p=pa),
                    rng.choice(build_b, n_b, p=pb) if n_b else np.empty(0, np.int64)]
        s_w = bs0[[bpos[int(q)] for q in win]]
        srv = s_w >= lam0
        aud = srv & (rng.random(W) < 0.1)
        lab = ~srv | aud
        hold = rng.random(W) < 0.2
        bsel = np.flatnonzero(~hold & lab)
        qs, wts = win[bsel], np.where(srv[bsel], 100, 10).astype(np.int64)
        half = rng.random(len(qs)) < 0.5
        prop = sim._greedy(qs[half], wts[half])
        new_hits = osim.hits_of_entries(D, prop, qs[~half], wts[~half], 10, T)
        cur_hits = osim.oracle_hits(sim.cache, qs[~half], wts[~half], 10, T)
        key = f"{eps}_{W}"
        res[f"dec_{key}"] = np.array([new_hits, cur_hits])
        if new_hits >= 1.1 * cur_hits:
            c1 = osim.Cache(D, 10, a.R, a.bits, sim.M)
            # free candidates are the current entries: copy cache 0 first
            c1.replace(dict(sim.cache.entries))
            c1.replace(sim._greedy(qs, wts))
            c1.set_p_to_fill()
            s1, y1 = arrays(c1, sim.model, ev)
            hq = win[hold & lab]
            hs, hy = arrays(c1, sim.model, hq)
            hw = np.where(s_w[hold & lab] >= lam0, 10.0, 1.0)
            hall, _ = arrays(c1, sim.model, win[hold])
            res.update({f"s1_{key}": s1, f"y1_{key}": y1, f"cal1s_{key}": hs,
                        f"cal1y_{key}": hy, f"cal1w_{key}": hw, f"ref1_{key}": hall})
        logger.info(f"{prep} eps {eps} W {W}: shadow new {new_hits} vs current {cur_hits}")
    np.savez(out / f"{prep}.npz", **res)
    logger.info(f"wrote {out / prep}.npz ({len(pos)} eval queries)")


#-------------------------------------------------------------------------------
# replay


def configs():
    out = []
    for det, resp, boost, rebuild in itertools.product(
            ("none", "strict", "loose1", "loose5", "loose20"),
            ("reset", "grace2"), (False, True), (False, True)):
        if det == "none" and (resp != "reset" or boost):
            continue
        out.append({"det": det, "resp": resp, "boost": boost, "rebuild": rebuild})
    return out


def name_of(c):
    return f"{c['det']}_{c['resp']}{'_boost' if c['boost'] else ''}{'_rebuild' if c['rebuild'] else ''}"


def replay(c, z, q_idx, u, P, eps, rng):
    from scipy.stats import ks_2samp
    from statistics import NormalDist
    from simulations.approximate_hits.threshold_selection import select_weighted, threshold_grid
    n = len(q_idx)
    s_cur, y_cur = z["s0"][q_idx], z["y0"][q_idx]
    cal = [(0, z["cal0_s"], z["cal0_y"], np.ones(len(z["cal0_s"])))]
    ref = [(0, z["cal0_s"])]
    aud_t, aud_y, aud_w = [], [], []
    lam, lam_hold, grace_left = None, None, 0
    j = 0
    target, boost_left = 30, 0
    rate = 0.1
    alarm_at, checks, cache_id = None, [], "0"
    log = []

    def calibrate(t):
        nonlocal cal, ref
        cal = [x for x in cal if x[0] > t - CAL_SPAN]
        ref = [x for x in ref if x[0] > t - CAL_SPAN]
        if not cal:
            return None
        s = np.concatenate([x[1] for x in cal])
        if len(s) < 2:
            return None
        return select_weighted(s, np.concatenate([x[2] for x in cal]),
                               np.concatenate([x[3] for x in cal]), threshold_grid(s), T, 0.05)
    lam = calibrate(0)
    alarms = rebuilds = audits = 0
    for b in range(n // B):
        lo, hi = b * B, (b + 1) * B
        s, y = s_cur[lo:hi], y_cur[lo:hi]
        use = lam_hold if grace_left > 0 else lam
        srv = s >= use if use is not None else np.zeros(B, bool)
        aud = srv & (u[lo:hi] < rate)
        audits += int(aud.sum())
        log.append((int(srv.sum()), float(y[srv].sum())))
        lab = ~srv | aud
        cal.append((hi, s[lab], y[lab], np.where(srv[lab], 1 / rate, 1.0)))
        ref.append((hi, s))
        idx = np.flatnonzero(aud) + lo
        aud_t.append(idx)
        aud_y.append(y[idx - lo])
        aud_w.append(np.full(len(idx), 1 / rate))
        tgt = 100 if boost_left > 0 else target
        boost_left = max(boost_left - 1, 0)
        rate = float(np.clip(tgt / max(srv.sum(), 1), 0.1, 1.0))
        if grace_left > 0:
            grace_left -= 1
        fired = False
        if c["det"] != "none":
            j += 1
            lv = (DELTA / 2 / (j * (j + 1)) if c["det"] == "strict"
                  else {"loose1": 0.01, "loose5": 0.05, "loose20": 0.2}[c["det"]])
            allsc = np.concatenate([x[1] for x in ref])
            rec, base = allsc[-2000:], allsc[:-2000]
            if len(rec) == 2000 and len(base) >= 1000 and \
                    ks_2samp(rec, base, method="asymp").pvalue < lv:
                fired = True
            if not fired:
                t_all = np.concatenate(aud_t)
                ya, wa = np.concatenate(aud_y), np.concatenate(aud_w)
                zq = NormalDist().inv_cdf(1 - lv / len(SCAN))
                for wq in SCAN:
                    m = t_all >= hi - wq
                    if m.sum() < 30:
                        continue
                    W_ = wa[m].sum()
                    mu = (wa[m] * ya[m]).sum() / W_
                    ne = W_ ** 2 / (wa[m] ** 2).sum()
                    sd = np.sqrt((wa[m] * (ya[m] - mu) ** 2).sum() / W_ * ne / max(ne - 1, 1))
                    if mu + zq * sd / np.sqrt(ne) < T:
                        fired = True
                        break
        if fired:
            alarms += 1
            if grace_left > 0:
                # an alarm during grace is evidence of harm: end grace, fresh only
                lam_hold, grace_left = None, 0
            elif c["resp"].startswith("grace"):
                lam_hold, grace_left = use, int(c["resp"][5:])
            cal, ref = [], []
            aud_t, aud_y, aud_w = [], [], []
            j = 0
            if c["boost"]:
                boost_left = 5
            alarm_at = hi
            checks = [hi + w for w in WINDOWS] if c["rebuild"] else []
        elif checks and hi >= checks[0]:
            W = WINDOWS[len(WINDOWS) - len(checks)]
            checks.pop(0)
            mix = eps if alarm_at >= P else 0.0
            key = f"{mix}_{W}"
            if f"s1_{key}" in z and cache_id == "0":
                s_cur = np.r_[s_cur[:hi], z[f"s1_{key}"][q_idx[hi:]]]
                y_cur = np.r_[y_cur[:hi], z[f"y1_{key}"][q_idx[hi:]]]
                cal = [(hi, z[f"cal1s_{key}"], z[f"cal1y_{key}"], z[f"cal1w_{key}"])]
                ref = [(hi, z[f"ref1_{key}"])]
                aud_t, aud_y, aud_w = [], [], []
                j, cache_id, checks = 0, key, []
                rebuilds += 1
        lam = calibrate(hi)
    srv = np.array([x[0] for x in log])
    ret = np.array([x[1] for x in log])
    pb = P // B
    post_srv, post_ret = srv[pb:], ret[pb:]
    win = [(post_srv[i:i + 5].sum(), post_ret[i:i + 5].sum()) for i in range(0, len(post_srv), 5)]
    wr = [r / s_ for s_, r in win if s_ > 0]
    return {"pre_hit": srv[:pb].sum() / (pb * B), "post_hit": post_srv.sum() / len(post_srv) / B,
            "post_ret": post_ret.sum() / max(post_srv.sum(), 1),
            "below": float(np.mean([w < T for w in wr])) if wr else 0.0,
            "worst": min(wr) if wr else None, "alarms": alarms, "rebuilds": rebuilds,
            "audits": audits / n * 1000}


def task(args):
    prep, eps, rep, P, H = args
    z = _G[prep]
    rng = np.random.default_rng([rep, int(eps * 1000), P, 7])
    na, nb = len(z["eval_a"]), len(z["eval_b"])
    fa = z["freq"][:na] / z["freq"][:na].sum()
    fb = z["freq"][na:] / z["freq"][na:].sum()
    pre = rng.choice(na, P, p=fa)
    n_b = rng.binomial(H, eps)
    post = np.r_[rng.choice(na, H - n_b, p=fa), na + rng.choice(nb, n_b, p=fb)]
    q_idx = np.r_[pre, rng.permutation(post)]
    u = rng.random(len(q_idx))
    return eps, rep, [replay(c, z, q_idx, u, P, eps, rng) for c in configs()]


def run(preps, out, R, P, H, workers):
    import multiprocessing as mp
    for prep in preps:
        _G[prep] = dict(np.load(out / f"{prep}.npz"))
        tasks = [(prep, eps, rep, P, H) for eps in EPS_GRID for rep in range(R)]
        res = {}
        with mp.get_context("fork").Pool(workers) as pool:
            for i, (eps, rep, o) in enumerate(pool.imap_unordered(task, tasks, chunksize=1)):
                res[(eps, rep)] = o
                if (i + 1) % 50 == 0:
                    logger.info(f"{prep}: {i + 1} / {len(tasks)} streams")
        table = {}
        for ci, c in enumerate(configs()):
            row = {}
            for eps in EPS_GRID:
                rs = [res[(eps, r)][ci] for r in range(R)]
                row[str(eps)] = {k: float(np.mean([x[k] for x in rs if x[k] is not None]))
                                 for k in rs[0]}
            table[name_of(c)] = {"config": c, "by_eps": row}
        dec = {k[4:]: _G[prep][k].tolist() for k in _G[prep] if k.startswith("dec_")}
        (out / f"{prep}_P{P}_response.json").write_text(json.dumps(
            {"P": P, "H": H, "R": R, "shadow": dec, "table": table}, indent=1))
        logger.info(f"wrote {out / prep}_P{P}_response.json")


def summary(path):
    d = json.load(open(path))
    print(f"== {path} (P = {d['P']}, H = {d['H']}, R = {d['R']})")
    print("shadow decisions (new / current oracle hits; installed iff >= 1.1x): " +
          " ".join(f"{k}: {v[0]}/{v[1]}" for k, v in d["shadow"].items()))
    eps = list(next(iter(d["table"].values()))["by_eps"])
    print(f"{'config':32s} " + " | ".join(f"eps {e}: hit  ret  <T  al rb" for e in eps) +
          " | pre-hit audits")
    for n, r in d["table"].items():
        b = r["by_eps"]
        cells = [f"{b[e]['post_hit']:.3f} {b[e]['post_ret']:.3f} {b[e]['below']:.2f} "
                 f"{b[e]['alarms']:3.1f} {b[e]['rebuilds']:3.1f}" for e in eps]
        print(f"{n:32s} " + " | ".join(cells) + f" | {b['0.0']['pre_hit']:.3f} "
              f"{b['1.0']['audits']:5.1f}")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--prepare", type=str, nargs="*", default=[])
    ap.add_argument("--run", type=str, nargs="*", default=[])
    ap.add_argument("--summary", type=str, nargs="*", default=[])
    ap.add_argument("--R", type=int, default=100)
    ap.add_argument("--P", type=int, default=20000)
    ap.add_argument("--H", type=int, default=40000)
    ap.add_argument("--workers", type=int, default=44)
    ap.add_argument("--out", type=str, default=str(RES / "response"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in args.prepare:
        prepare(p, out)
    if args.run:
        run(args.run, out, args.R, args.P, args.H, args.workers)
    for p in args.summary:
        summary(p)


if __name__ == "__main__":
    main()
