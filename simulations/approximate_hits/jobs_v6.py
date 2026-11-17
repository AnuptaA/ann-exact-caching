#!/usr/bin/env python3
"""Job lines of the v6 experiments (doc 9ah): every technique through the one driver (online_sim.py) on the
C++ core, workloads from the registry (workloads.py).

  --set tune     stage-1 / stage-2 search lists on the core (tune_core.py) for every timing cell
  --set timing   timed runs (composed: misses charged the request's measured DiskANN time, base node
                 cache; DiskANN-only arms from v5, copied): ours static (not on real logs), ours drift,
                 ours continuous greedy, LRU (our rule), QVCache (core, hnswlib mini-indexes), real Aker;
                 Aker / QVCache: the CANDIDATES best in the headline (up to 3 settings meeting the target,
                 by hit rate), so the timing plot can pick, in the timing setting itself, the best one that
                 meets the target (doc 9ah: settings picked in the headline did not always meet it in timing);
                 plus the papers' own settings where the headline has them
Results: results/tuning_core/<workload>/b<b>.json, results/timing_v6/<workload>/<tag>_b<b>.json.

  PYTHONPATH=. .venv/bin/python simulations/approximate_hits/jobs_v6.py --set timing > /tmp/mx/v6_timing.txt
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from simulations.approximate_hits.workloads import WORKLOADS, RES          # noqa: E402

S = "simulations/approximate_hits"
PY = ".venv-core/bin/python -u"
EVERY = 50
AKER_LIB = "simulations/approximate_hits/aker_real:/home/jialind/third_party/aker/build/lib"
AKER_PROX_LIB = "/home/jialind/third_party/aker_prox/build/lib:simulations/approximate_hits/aker_real"
T, TOL = 0.9, 0.01


def v5_dir(w):
    return RES / "timing_v5" / (w.backend or w.prep)


def tuning_file(w, b):
    return RES / "tuning_core" / w.name / f"b{b:g}.json"


def tune_lines():
    out = []
    for w in WORKLOADS.values():
        for b in w.budgets:
            st = w.state_dir / f"state_b{b:g}.npz"
            if not st.exists():
                continue
            lam = ""
            old = w.state_dir / f"stage1_b{b:g}.json"          # cells whose state has no cutoff (doc 9af)
            if old.exists():
                c = json.loads(old.read_text()).get("cutoffs") or []
                import numpy as np
                if c and not np.isfinite(float(np.load(st)["lam"])):
                    lam = " --lam " + ",".join(f"{x:g}" for x in c)
            f = tuning_file(w, b)
            out.append(f"mkdir -p {f.parent} && [ -f {f} ] || PYTHONPATH=. {PY} {S}/tune_core.py --workload {w.name} "
                       f"--budget {b:g}{lam} > {f.with_suffix('.txt')} 2>&1")
    return out


def _cands(prep, budget, kind):
    """(name, hit %, retention) of the headline's settings of Aker (real) or QVCache."""
    from simulations.approximate_hits.plot_headline import sim_dir, sim_stats
    out = []
    if kind == "aker":
        for f in sorted((RES / "aker_real" / prep / f"trace_b{budget:g}").glob("*.json")):
            s = json.loads(f.read_text())["summary"]
            out.append((f.stem, 100 * s["hit_rate"], s["retention_vs_backend"]))
    else:
        d0 = sim_dir(prep, "trace" if not prep.startswith("esci") else "random", budget, 0)
        for f in sorted(d0.glob("qvcache_*.json")) if d0.exists() else []:
            r = sim_stats(f)
            if r:
                out.append((f.stem, r[0], r[1]))
    return out


def candidates(prep, budget, kind, n=3):
    c = _cands(prep, budget, kind)
    ok = sorted([x for x in c if x[2] >= T - TOL], key=lambda x: -x[1])
    if ok:
        return [x[0] for x in ok[:n]]
    return [max(c, key=lambda x: x[2])[0]] if c else []


def aker_args(name):
    import re
    m = re.fullmatch(r"prox([0-9.]+)_d(\d+)", name)
    if m:
        return f"--aker-real --aker-real-prox {m[1]} --aker-real-delta {m[2]}", AKER_PROX_LIB
    m = re.fullmatch(r"at([0-9.]+)(?:_al([0-9.]+))?_d(\d+)", name)
    if not m:
        raise SystemExit(f"unknown Aker setting {name}")
    al = f" --aker-real-al {m[2]}" if m[2] else ""
    return f"--aker-real --aker-real-at {m[1]}{al} --aker-real-delta {m[3]}", AKER_LIB


def qv_args(name):
    import re
    m = re.fullmatch(r"qvcache_D([0-9.]+)_m(\d+)_r(\d+)x(\d+)", name)
    if not m:
        raise SystemExit(f"unknown QVCache setting {name}")
    return f"--qv-D {m[1]} --qv-mini {m[2]} --qv-dims {m[3]} --qv-buckets {m[4]}"


def timing_lines():
    out = []
    for w in WORKLOADS.values():
        arm = v5_dir(w) / f"backend_s{EVERY}.json"
        if not arm.exists():
            continue
        o = RES / "timing_v6" / w.name
        prep_flags = (f"--prep {w.prep_dir} --stream {w.stream} --seed 0 --backend-file {w.answers_file} "
                      f"--backend-target {w.target} --cache-impl core --core-threads 4 "
                      f"--timing-compose {v5_dir(w)}/backend_s{EVERY}_lat.npy --timing-every {EVERY}")
        # the DiskANN-only arms (base node cache; measured once per workload, quiet): from v5
        out.append(f"mkdir -p {o}/logs && cp -n {arm} {v5_dir(w)}/backend_s{EVERY}_lat.npy {o}/")
        for b in w.budgets:
            tf = tuning_file(w, b)
            jobs = []
            ours = ("--shell-scope balls --shell-balls 10 "
                    f"$(python3 -c \"import json; d=json.load(open('{tf}')); print('--core-ef1', d['stage1']['ef'], '--core-ef2', d['stage2']['ef'])\")")
            if not w.real_log:
                jobs.append(("ours", f"--method static --detect {ours}", ""))
            jobs.append(("oursdrift", f"--method greedy --detect score --response rebuild {ours}", ""))
            jobs.append(("oursonline", f"--method ogreedy --og-evict utility --detect {ours}", ""))
            jobs.append(("lru", f"--method lru --k-lru 20 --detect {ours}", ""))
            for name in candidates(w.prep, b, "qvcache"):
                jobs.append((name, f"--method qvcache {qv_args(name)}", ""))
            for name in candidates(w.prep, b, "aker"):
                a, lib = aker_args(name)
                jobs.append((f"aker_{name}", f"--method aker {a}", lib))
            if w.paper_settings:
                paper = RES / "aker_real" / w.prep / f"trace_b{b:g}" / "at0.25_d0.json"
                if paper.exists():
                    a, lib = aker_args("at0.25_d0")
                    jobs.append(("akerpaper_at0.25_d0", f"--method aker {a}", lib))
            for tag, extra, lib in jobs:
                d = o / f"{tag}_b{b:g}"
                js = d / ("static.json" if extra.startswith("--method static") else "")
                env = f"LD_LIBRARY_PATH={lib} " if lib else ""
                done = f"{d}/DONE"
                out.append(f"[ -f {done} ] || {{ [ -f {tf} ] && PYTHONPATH=. {env}{PY} {S}/online_sim.py {prep_flags} "
                           f"--budget {b:g} {extra} --out {d} > {o}/logs/{tag}_b{b:g}.txt 2>&1 && touch {done}; }}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["tune", "timing"], required=True)
    a = ap.parse_args()
    print("\n".join(tune_lines() if a.set == "tune" else timing_lines()))


if __name__ == "__main__":
    main()
