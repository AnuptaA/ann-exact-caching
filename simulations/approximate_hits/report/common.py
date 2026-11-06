#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Helpers shared by the report commands."""

import json
from pathlib import Path

# overridable with `report --results-dir`
RESULTS = Path(__file__).resolve().parents[1] / "results"

#-------------------------------------------------------------------------------


def load_json(path):
    """Parsed JSON file, or None if it does not exist."""
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def target_rows(path, target):
    """Rows of the results entry for `target` in a results-style JSON, else None."""
    data = load_json(path)
    if data is None:
        return None
    for t in data["results"]:
        if abs(t["target"] - target) < 1e-9:
            return t["rows"]
    return None


def nearest(rows, size, tol=None):
    """Row whose cache_frac is nearest `size`; None if farther than tol * size."""
    row = min(rows, key=lambda r: abs(r["cache_frac"] - size))
    if tol is not None and abs(row["cache_frac"] - size) >= tol * size:
        return None
    return row
