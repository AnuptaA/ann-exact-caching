#!/usr/bin/env python3
"""Summarize eval_detectors.py results: per detector, false-alarm rate (any
alarm before the shift, eps = 0 run), detection rate / median delay per eps,
and audits per 1000 queries.

  python eval_detectors_summary.py results/detectors/esci1m_P20000_detectors.json
      [--filter ret_] [--eps 0.1 0.3 0.5 1.0]
"""

import sys
import json
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--filter", type=str, default="")
    ap.add_argument("--eps", type=str, nargs="+", default=["0.1", "0.3", "0.5", "0.7", "1.0"])
    args = ap.parse_args()
    for path in args.files:
        d = json.load(open(path))
        print(f"\n== {path} (P = {d['P']}, H = {d['H']}, R = {d['R']})")
        print("served retention / hit rate of the post-shift mix at the frozen lambda: " +
              "  ".join(f"eps {e}: {d['harm'][e][0]:.3f} / {d['harm'][e][1]:.3f}"
                        for e in ["0.0"] + args.eps))
        print(f"{'detector':40s} {'false':>5s} | " +
              " | ".join(f"eps {e}: det  delay" for e in args.eps) + " | audits/1k")
        for name, r in d["table"].items():
            if args.filter not in name:
                continue
            b = r["by_eps"]
            cells = []
            for e in args.eps:
                v = b[e]
                dl = "  -- " if v["median_delay"] is None else f"{v['median_delay']:6.0f}"
                cells.append(f"      {v['detect']:4.2f} {dl}")
            print(f"{name:40s} {b['0.0']['false_alarm']:5.3f} | " + " | ".join(cells) +
                  f" | {b['1.0']['audits_per_1000']:5.1f}")


if __name__ == "__main__":
    sys.exit(main())
