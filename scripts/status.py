#!/usr/bin/env python3
"""Runs done, missing and failed per question and system, from the local result directories.

  python -m scripts.status --jobs jobs/
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from recal.analysis import figures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=Path, default=Path("jobs"))
    a = ap.parse_args(argv)
    table = json.loads((a.jobs / "matrix_runs.json").read_text())
    counts: dict = {}
    for entry in table.values():
        key = (entry["cells"][0]["experiment"], entry["system"])
        counts.setdefault(key, Counter())[figures.run_state(entry)] += 1
    total = Counter()
    for (question, system), c in sorted(counts.items()):
        total.update(c)
        print(f"{question} {system:24} done {c['done']:4d} missing {c['missing']:4d} failed {c['failed']:3d}")
    print(f"total: done {total['done']}, missing {total['missing']}, failed {total['failed']}")
    return 1 if total["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
