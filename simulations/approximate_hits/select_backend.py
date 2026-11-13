#!/usr/bin/env python3
"""
Choose each workload's DiskANN setting from the quiet latency probes (doc 9u): over the candidate
indexes of its database, every (search list, beam width) probed, the one with the lowest mean
single-query latency whose recall@10 (5,000 queries) reaches the target (0.95; 0.9 when 0.95 would
exceed --max-ms). Writes results/backends/<prep>/choice.json.

  .venv/bin/python simulations/approximate_hits/select_backend.py
"""

import json
import argparse
from pathlib import Path

RES = Path("simulations/approximate_hits/results")
PROBE = RES / "latency_probe"
# workload -> candidate probes (each probe = one index measured on that database's queries)
CANDIDATES = {
    **{f"spacev-aker-{z}": ["spacev", "spacev-pq110"] for z in ("0.3", "0.6", "0.99")},
    "spacev-real": ["spacev-real", "spacev-real-pq110"],
    "esci1m": ["esci"],
    **{f"triviaqa-ip-{z}": ["triviaqa-base", "triviaqa-pq768", "triviaqa-r128", "triviaqa-aker"]
       for z in ("0.3", "0.6", "0.99")},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=float, nargs="+", default=[0.95, 0.9])
    ap.add_argument("--max-ms", type=float, default=100.0)
    args = ap.parse_args()
    for prep, probes in CANDIDATES.items():
        rows = []
        for name in probes:
            f = PROBE / f"{name}.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            for g in d["grid"]:
                if "recall_5000" in g:
                    rows.append(dict(g, probe=name, index=d["index"]))
        if not rows:
            print(f"{prep}: no probe yet")
            continue
        choice = None
        for tgt in args.targets:
            ok = [r for r in rows if r["recall_5000"] >= tgt]
            if not ok:
                continue
            best = min(ok, key=lambda r: r["mean_ms"])
            choice = dict(target=tgt, index_dir=best["index"], probe=best["probe"], L=best["L"], W=best["W"],
                          recall_5000=best["recall_5000"], mean_ms=best["mean_ms"], p99_ms=best["p99_ms"])
            if best["mean_ms"] <= args.max_ms:
                break
        if choice is None:
            print(f"{prep}: no setting reaches {min(args.targets)}")
            continue
        out = RES / "backends" / prep
        out.mkdir(parents=True, exist_ok=True)
        (out / "choice.json").write_text(json.dumps(choice, indent=1))
        print(f"{prep}: {choice}")


if __name__ == "__main__":
    main()
