"""Robustness experiments: burst marking, budget and history grids, K streams."""

import numpy as np

BUDGETS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
HISTORIES = (10_000, 30_000, 50_000)
STREAM_LENGTH = 100_000
K_VALUES = (5, 10, 50, 100)


def burst_mask(keys: np.ndarray, block: int = 1000, min_count: int = 5) -> np.ndarray:
    """A request is in a burst when its query occurs at least min_count times in its block of requests."""
    keys = np.asarray(keys)
    out = np.zeros(len(keys), dtype=bool)
    for s in range(0, len(keys), block):
        _, inverse, counts = np.unique(keys[s:s + block], return_inverse=True, return_counts=True)
        out[s:s + block] = counts[inverse] >= min_count
    return out


def split_by_mask(cols: dict, mask: np.ndarray) -> dict:
    """Per-request records restricted to the requests with mask true and false (aligned with cols['index'])."""
    from recal.analysis.rates import rates
    index = cols["index"]
    sel = mask[index]
    return {"burst": rates({k: v[sel] for k, v in cols.items()}),
            "other": rates({k: v[~sel] for k, v in cols.items()})}


def grid_lines(prep: str, stream: str, name: str, budgets=BUDGETS, histories=HISTORIES, seeds=(0, 1),
               extra: str = "", method: str = "recal") -> list[str]:
    """ReCal cells over the budget grid (history 30K) and the history grid (budget 5%), per seed."""
    cells = [(b, 30_000) for b in budgets] + [(0.05, h) for h in histories if h != 30_000]
    out = []
    for b, h in cells:
        frac = h / STREAM_LENGTH
        for seed in seeds:
            d = f"simulations/approximate_hits/results/q4/{name}/{method}_b{b:g}_h{h}_s{seed}"
            out.append(f"[ -f {d}/{method}.json ] || {{ mkdir -p {d} && PYTHONPATH=. .venv-core/bin/python -u -m "
                       f"recal.cells --method {method} --prep {prep} --stream {stream} --budget {b:g} "
                       f"--history {h} --history-frac {frac:g} --seed {seed} {extra} --out {d} "
                       f"> {d}/log.txt 2>&1; }}  # recal.cells")
    return out


def k_lines(prep: str, stream: str, name: str, budget: float = 0.05, seeds=(0, 1), k_values=K_VALUES,
            extra: str = "") -> list[str]:
    """Fixed K streams and the mixed stream."""
    out = []
    for label, flags in [(f"k{k}", f"--k {k}") for k in k_values] + [("mixed", "--k 10 --k-mixed")]:
        for seed in seeds:
            d = f"simulations/approximate_hits/results/q4/{name}/{label}_s{seed}"
            out.append(f"[ -f {d}/recal.json ] || {{ mkdir -p {d} && PYTHONPATH=. .venv-core/bin/python -u -m "
                       f"recal.cells --method recal --prep {prep} --stream {stream} --budget {budget:g} {flags} "
                       f"--history-frac 0.3 --seed {seed} {extra} --out {d} > {d}/log.txt 2>&1; }}  # recal.cells")
    return out


def shift_lines(prep: str, name: str, shifts=("abrupt", "gradual", "correlated"), budget: float = 0.05,
                seeds=(0, 1, 2, 3, 4), signals=(True, False), extra: str = "") -> list[str]:
    """Shift timelines: the prep's stream named after the shift, with and without the rebuild signals."""
    out = []
    for kind in shifts:
        for on in signals:
            tag = "signals" if on else "periodic"
            for seed in seeds:
                d = f"simulations/approximate_hits/results/q4/{name}/{kind}_{tag}_s{seed}"
                flag = "" if on else "--no-rebuild-signals"
                out.append(f"[ -f {d}/recal.json ] || {{ mkdir -p {d} && PYTHONPATH=. .venv-core/bin/python -u -m "
                           f"recal.cells --method recal --prep {prep} --stream {kind} --budget {budget:g} "
                           f"--history-frac 0.3 --seed {seed} {flag} {extra} --out {d} > {d}/log.txt 2>&1; }}"
                           "  # recal.cells")
    return out


def burst_lines(prep: str, stream: str, name: str, budget: float = 0.05, seeds=(0, 1), extra: str = "") -> list[str]:
    """Burst experiment: the stream with and without the mini cache (periods are marked from the records)."""
    out = []
    for label, flag in (("mini", ""), ("nomini", "--no-mini-cache")):
        for seed in seeds:
            d = f"simulations/approximate_hits/results/q4/{name}/burst_{label}_s{seed}"
            out.append(f"[ -f {d}/recal.json ] || {{ mkdir -p {d} && PYTHONPATH=. .venv-core/bin/python -u -m "
                       f"recal.cells --method recal --prep {prep} --stream {stream} --budget {budget:g} "
                       f"--history-frac 0.3 --seed {seed} {flag} {extra} --out {d} > {d}/log.txt 2>&1; }}"
                       "  # recal.cells")
    return out


def update_lines(prep: str, stream: str, name: str, schedules: dict, budget: float = 0.05, seeds=(0, 1),
                 methods=("recal", "aker"), extra: str = "") -> list[str]:
    """Data-update cells. schedules: label (for example targeted5) -> update schedule npz."""
    out = []
    for label, path in schedules.items():
        for method in methods:
            for seed in seeds:
                d = f"simulations/approximate_hits/results/q4/{name}/updates_{label}_{method}_s{seed}"
                out.append(f"[ -f {d}/{method}.json ] || {{ mkdir -p {d} && PYTHONPATH=. .venv-core/bin/python -u -m "
                           f"recal.cells --method {method} --prep {prep} --stream {stream} --budget {budget:g} "
                           f"--backend exact --history-frac 0.3 --seed {seed} --updates {path} {extra} --out {d} "
                           f"> {d}/log.txt 2>&1; }}  # recal.cells")
    return out
