#!/usr/bin/env python3
"""
Job lines for the timing plot (eval_timing_online.py; context doc 9k): every technique in front of
the same DiskANN (search list tuned to recall@10 0.95), SPACEV Zipf 0.3 / 0.6 / 0.99, cache budgets
1% and 10%. Aker and QVCache run their best setting meeting the target in the headline results
(plot_headline.collect: best hit rate among settings with mean retention of hits >= T - 0.01;
else the highest-retention setting). Ours and LRU start from the warm-up state of
online_sim --save-state with DiskANN's answers (results/timing_v2/<prep>/state_b<b>.npz); LRU runs
twice: cache maintenance on the request path (sync) and as background work.

  PYTHONPATH=. .venv/bin/python simulations/approximate_hits/timing_jobs.py > /tmp/mx/timing.txt
"""

import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from simulations.approximate_hits.plot_headline import collect  # noqa: E402

S = "simulations/approximate_hits"
R = f"{S}/results"
AKER_LIB = ("/home/jialind/third_party/aker/build/lib:/home/jialind/third_party/boost-install/lib:"
            "/home/jialind/third_party/faiss-install/lib64")
AKER_PROX_LIB = "/home/jialind/third_party/aker_prox/build/lib:" + AKER_LIB


def aker_args(name):
    """Real-Aker result name -> (eval_timing_online args, LD_LIBRARY_PATH)."""
    m = re.fullmatch(r"prox([0-9.]+)_d(\d+)", name)
    if m:
        return f"--prox-frac {m[1]} --aker-delta {m[2]}", AKER_PROX_LIB
    m = re.fullmatch(r"at([0-9.]+)(?:_al([0-9.]+))?_d(\d+)", name)
    if not m:
        raise SystemExit(f"unknown Aker setting {name}")
    al = f" --alpha-loosen {m[2]}" if m[2] else ""
    return f"--alpha-tighten {m[1]}{al} --aker-delta {m[3]}", AKER_LIB


def qv_args(name):
    m = re.fullmatch(r"qvcache_D([0-9.]+)_m(\d+)_r(\d+)x(\d+)", name)
    if not m:
        raise SystemExit(f"unknown QVCache setting {name}")
    return f"--qv-D {m[1]} --qv-mini {m[2]} --qv-dims {m[3]} --qv-buckets {m[4]}"


PROBE = f"{R}/latency_probe"


def probe_lines():
    """Quiet latency probe (latency_probe.py): search list x beam width per candidate index (doc 9u)."""
    import os
    I = f"{R}/backends/index_study"
    sp = [100, 128, 160, 200, 256, 320, 400, 512]
    big = [128, 200, 320, 512, 800, 1000, 1250, 1600, 2000]
    jobs = [("spacev", "spacev-aker-0.6", f"{R}/backends/spacev10m/diskann", sp),
            ("spacev-pq110", "spacev-aker-0.6", f"{I}/spacev_pq110/index", sp),
            ("spacev-real", "spacev-real", f"{R}/backends/spacev10m/diskann", big),
            ("spacev-real-pq110", "spacev-real", f"{I}/spacev_pq110/index", big),
            ("esci", "esci1m", f"{R}/backends/esci1m/diskann", [48, 64, 80, 100, 128, 160, 200]),
            ("triviaqa-base", "triviaqa-ip-0.6", f"{R}/backends/sphereip10m-pq256/diskann", big),
            ("triviaqa-pq768", "triviaqa-ip-0.6", f"{I}/pq768/index", big),
            ("triviaqa-r128", "triviaqa-ip-0.6", f"{I}/r128/index", big),
            ("triviaqa-aker", "triviaqa-ip-0.6", f"{I}/aker_tqa/index", big),
            # new workloads (11-13, doc 9x / 9y): one DiskANN each (degree 64, build list 100, 256-byte PQ)
            ("tripclick", "tripclick-ip", f"{R}/backends/tripclick-ip/diskann", big),
            ("medrag", "medrag-ip", f"{R}/backends/medrag-ip/diskann", big)]
    out = []
    for name, prep, index, eff in jobs:
        if not os.path.exists(f"{index}/ann_disk.index"):
            continue
        js = f"{PROBE}/{name}.json"
        out.append(f"mkdir -p {PROBE}/logs && [ -f {js} ] || .venv-ann/bin/python -u {S}/latency_probe.py "
                   f"--prep {R}/online/{prep} --index {index} --efforts {' '.join(map(str, eff))} --wait-quiet 30 "
                   f"--out {js} > {PROBE}/logs/{name}.txt 2>&1")
    return out


def target_of(prep):
    """The recall target chosen for a workload (select_backend.py), 0.95 by default."""
    import json as _json
    try:
        return f"{_json.load(open(f'{R}/backends/{prep}/choice.json'))['target']:g}"
    except Exception:
        return "0.95"


def beam_for(prep, tgt="0.95"):
    """Beam width for a prep's timing runs: at its tuned search list, the fastest beam width whose
    results equal the beam-2 ones on >= 99% of the probe queries (the replay's stored answers are
    beam-2 searches) and whose recall is within .005 of beam 2's; 2 without a probe."""
    import json as _json
    name = "esci" if prep.startswith("esci") else ("spacev-real" if prep == "spacev-real" else
                                                   ("spacev" if prep.startswith("spacev") else None))
    try:
        L = int(_json.load(open(f"{R}/backends/{prep}/diskann.json"))["targets"][tgt]["effort"])
        grid = _json.load(open(f"{PROBE}/{name}.json"))["grid"]
    except Exception:
        return 2
    rows = [g for g in grid if g["L"] == L]
    b2 = [g for g in rows if g["W"] == 2]
    if not b2:
        return 2
    ok = [g for g in rows if g["same_as_beam2"] >= 0.99 and g["recall"] >= b2[0]["recall"] - 0.005]
    return min(ok, key=lambda g: g["mean_ms"])["W"] if ok else 2


def lines(prep, b, every, techs=("ours", "lru", "aker", "qvcache"), out_root="timing_v4", stream="trace"):
    """Full-stream runs (doc 9m): the whole stream after warm-up, every `every`-th request timed
    live, cache updates as background work for every technique; ours static (no recalibration /
    audits), LRU recalibrating, seeded with the warm-up's labels."""
    pick = collect(prep, stream, b, True)
    state = f"{R}/{out_root}/{prep}/state_b{b:g}.npz"
    out = f"{R}/{out_root}/{prep}"
    base = (f".venv-ann/bin/python -u {S}/eval_timing_online.py --prep {R}/online/{prep} "
            f"--budget {b:g} --sample-every {every} --maintenance background --wait-quiet 30 --stream {stream} "
            f"--target-recall {target_of(prep)}")
    jobs = []
    if "ours" in techs:
        jobs.append((f"ours_s{every}_b{b:g}", f"--method ours --state {state}", ""))
    if "ours_drift" in techs:   # drift detection + shadow rebuild, run live (doc 9ac)
        jobs.append((f"oursdrift_s{every}_b{b:g}", f"--method ours_drift --state {state}", ""))
    if "lru" in techs:      # CAL_DEBUG: dump the labels if the cutoff is ever lost (doc 9o)
        jobs.append((f"lru_s{every}_b{b:g}", f"--method lru --state {state}", "",
                     f"CAL_DEBUG={out}/lru_s{every}_b{b:g}_caldbg.npz "))
    if "aker" in techs and pick.get("aker"):
        a, lib = aker_args(pick["aker"]["name"])
        jobs.append((f"aker_{pick['aker']['name']}_s{every}_b{b:g}", f"--method aker {a}", lib))
    if "aker_paper" in techs and pick.get("aker_paper"):     # the papers' own settings (doc 9q)
        a, lib = aker_args(pick["aker_paper"]["name"])
        jobs.append((f"akerpaper_{pick['aker_paper']['name']}_s{every}_b{b:g}", f"--method aker {a}", lib))
    if "qvcache_paper" in techs and pick.get("qvcache_paper"):
        jobs.append((f"qvpaper_{pick['qvcache_paper']['name']}_s{every}_b{b:g}",
                     f"--method qvcache {qv_args(pick['qvcache_paper']['name'])}", ""))
    if "qvcache" in techs and pick.get("qvcache"):
        jobs.append((f"{pick['qvcache']['name']}_s{every}_b{b:g}",
                     f"--method qvcache {qv_args(pick['qvcache']['name'])}", ""))
    res = []
    for job in jobs:
        tag, extra, lib = job[:3]
        env = (f"LD_LIBRARY_PATH={lib} " if lib else "") + (job[3] if len(job) > 3 else "")
        js = f"{out}/{tag}.json"
        res.append(f"mkdir -p {out}/logs && [ -f {js} ] || {env}{base} {extra} --out {js} "
                   f"> {out}/logs/{tag}.txt 2>&1")
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", action="store_true", help="the first batch (2,000 requests after warm-up)")
    ap.add_argument("--set", choices=["matrix", "lru_retime", "retime", "sizes", "final", "probe", "new"], default="matrix",
                    help="lru_retime: LRU again after the empty-search fix (doc 9n); sizes: the extra "
                         "cache sizes 0.3% and 3%, every technique, 1 in 20")
    a = ap.parse_args()
    if a.v2:
        return main_v2()
    if a.set == "lru_retime":
        out = []
        for z in ("0.99", "0.6", "0.3"):
            for b in (0.01, 0.1):
                out += lines(f"spacev-aker-{z}", b, 20, techs=("lru",))
        out += lines("spacev-aker-0.6", 0.01, 10, techs=("lru",))
        print("\n".join(out))
        return
    if a.set == "retime":       # ours and LRU after the served-label / in-cache search change (doc 9o)
        out = []
        for z in ("0.99", "0.6", "0.3"):
            for b in (0.01, 0.1):
                out += lines(f"spacev-aker-{z}", b, 20, techs=("ours", "lru"))
        print("\n".join(out))
        return
    if a.set == "probe":
        print("\n".join(probe_lines()))
        return
    if a.set == "final":        # v4 (doc 9u): tuned DiskANN (search list x beam width); ordered so the
        out = []                # plot fills in early: (11-14) every workload's 1% / 10% main cells
        main = ("ours", "ours_drift", "lru", "aker", "qvcache")   # first, then the papers' settings,
        paper = ("aker_paper", "qvcache_paper")                     # then 0.3% / 3%
        real_log = ("ours_drift", "lru", "aker", "qvcache")   # real logs: drift handling only (11-14)
        work = [(f"spacev-aker-{z}", "trace", True) for z in ("0.99", "0.6", "0.3")] + \
            [("esci1m", "random", True)] + [(f"triviaqa-ip-{z}", "trace", False) for z in ("0.99", "0.6", "0.3")] + \
            [("spacev-real", "trace", False)]
        for prep, stream, _ in work:
            for b in (0.01, 0.1):
                out += lines(prep, b, 20, stream=stream, techs=real_log if prep == "spacev-real" else main)
        for prep, stream, extra in work:
            if extra:
                for b in (0.01, 0.1):
                    out += lines(prep, b, 20, stream=stream, techs=paper)
        for prep, stream, extra in work:
            if extra:
                for b in (0.003, 0.03):
                    out += lines(prep, b, 20, stream=stream, techs=main)
        print("\n".join(out))
        return
    if a.set == "new":          # new workloads (doc 9x / 9y), once their backend is chosen and states exist
        import os
        out = []
        sizes = {"tripclick-ip": (0.01, 0.1, 0.003, 0.03),
                 "medrag-ip": (0.001, 0.0001, 0.0003, 0.003)}   # MedRAG: 0.3% already holds the working set
        for prep in ("tripclick-ip", "medrag-ip"):
            if not os.path.exists(f"{R}/backends/{prep}/choice.json"):
                continue
            for b in sizes[prep]:
                if os.path.exists(f"{R}/timing_v4/{prep}/state_b{b:g}.npz"):
                    out += lines(prep, b, 20, techs=("ours_drift", "lru", "aker", "qvcache") if prep == "tripclick-ip"
                                 else ("ours", "ours_drift", "lru", "aker", "qvcache"))
        print("\n".join(out))
        return
    if a.set == "sizes":
        out = []
        for z in ("0.99", "0.6", "0.3"):
            for b in (0.003, 0.03):
                out += lines(f"spacev-aker-{z}", b, 20)
        print("\n".join(out))
        return
    out = []
    for z in ("0.99", "0.6", "0.3"):             # the matrix, 1 in 20
        for b in (0.01, 0.1):
            out += lines(f"spacev-aker-{z}", b, 20)
    for every in (10, 5):                         # sample-rate check on two cells (ours, LRU)
        for z, b in (("0.6", 0.01), ("0.99", 0.1)):
            out += lines(f"spacev-aker-{z}", b, every, techs=("ours", "lru"))
    print("\n".join(out))


def main_v2():
    for z in ("0.99", "0.6", "0.3"):
        prep = f"spacev-aker-{z}"
        for b in (0.01, 0.1):
            pick = collect(prep, "trace", b, True)
            out = f"{R}/timing_v2/{prep}"
            state = f"{out}/state_b{b:g}.npz"
            base = (f".venv-ann/bin/python -u {S}/eval_timing_online.py --prep {R}/online/{prep} "
                    f"--budget {b:g} --n 2000 --wait-quiet 30")
            jobs = [(f"ours_b{b:g}", f"--method ours --state {state}", ""),
                    (f"lru_sync_b{b:g}", f"--method lru --state {state}", ""),
                    (f"lru_bg_b{b:g}", f"--method lru --maintenance background --state {state} "
                                       f"--warmup 21000", "")]
            if pick.get("aker"):
                a, lib = aker_args(pick["aker"]["name"])
                jobs.append((f"aker_{pick['aker']['name']}_b{b:g}", f"--method aker {a}", lib))
            if pick.get("qvcache"):
                jobs.append((f"{pick['qvcache']['name']}_b{b:g}",
                             f"--method qvcache {qv_args(pick['qvcache']['name'])}", ""))
            for tag, extra, lib in jobs:
                env = f"LD_LIBRARY_PATH={lib} " if lib else ""
                js = f"{out}/{tag}.json"
                print(f"mkdir -p {out}/logs && [ -f {js} ] || {env}{base} {extra} --out {js} "
                      f"> {out}/logs/{tag}.txt 2>&1")


if __name__ == "__main__":
    main()
