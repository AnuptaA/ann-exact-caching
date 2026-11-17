#!/usr/bin/env python3
"""Moves v5 runs made before the audit-accounting fix (doc 9af) that audited at least once on a timed
request to timing_v5/<prep>/audit_bug/, so run_timing_composed.py reruns them."""
import json
from pathlib import Path
n = 0
for f in Path("simulations/approximate_hits/results/timing_v5").glob("*/*_s50_b*.json"):
    if f.name.startswith("backend"):
        continue
    a = json.loads(f.read_text()).get("audits_timed") or {}
    if a.get("n", 0) > 0 and a.get("accounting") != 2:
        d = f.parent / "audit_bug"
        d.mkdir(exist_ok=True)
        f.rename(d / f.name)
        lat = f.with_name(f.stem + "_lat.npz")
        if lat.exists():
            lat.rename(d / lat.name)
        n += 1
print(f"moved {n} runs")
