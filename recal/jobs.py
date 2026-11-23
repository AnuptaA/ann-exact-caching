#!/usr/bin/env python3
"""Job lines (one shell command per line, each skipping itself when its result exists) for ReCal cells.

  python -m recal.jobs --prep simulations/approximate_hits/results/online/esci50k --stream random \\
      --budgets 0.01 0.05 0.1 --methods recal prototype aker qvcache proximity-p0.1 proximity-p1 > jobs.txt
  python simulations/approximate_hits/run_jobs.py jobs.txt --workers 2
"""

import argparse

PY = ".venv-core/bin/python"


def method_flags(token: str) -> tuple[str, str, str]:
    """(method, result file stem, flags) of a method token; proximity-p0.1 is Proximity at the 0.1 percentile."""
    if token.startswith("proximity-p"):
        pct = token[len("proximity-p"):]
        return "proximity", f"proximity_p{pct}", f"--prox-percentile {pct} --tag _p{pct}"
    return token, token, ""


def lines(prep: str, stream: str, budgets, methods, name: str, extra: str = "") -> list[str]:
    out = []
    for b in budgets:
        for m in methods:
            method, stem, flags = method_flags(m)
            d = f"simulations/approximate_hits/results/recal/{name}/{m}_b{b:g}"
            out.append(f"[ -f {d}/{stem}.json ] || {{ mkdir -p {d} && PYTHONPATH=. {PY} -u -m recal.cells "
                       f"--method {method} {flags} "
                       f"--prep {prep} --stream {stream} --budget {b:g} {extra} --out {d} "
                       f"> {d}/log.txt 2>&1; }}  # recal.cells online_sim.py")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--stream", default="random")
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    ap.add_argument("--methods", nargs="+", default=["recal", "prototype", "aker", "qvcache", "proximity-p0.1", "proximity-p1"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    name = a.name or a.prep.rstrip("/").split("/")[-1]
    print("\n".join(lines(a.prep, a.stream, a.budgets, a.methods, name, a.extra)))


if __name__ == "__main__":
    main()
