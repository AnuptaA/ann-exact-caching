#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Generic debug-output I/O helpers, shared across microbenchmark debug modules."""

import csv
from pathlib import Path

#-------------------------------------------------------------------------------

def write_txt(path: Path, lines: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for line in lines:
            f.write(line + '\n')

def write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
