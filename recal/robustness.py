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
        hist = "" if h == 30_000 else f"--history {h} --history-frac {frac:g} "
        for seed in seeds:
            d = f"simulations/approximate_hits/results/q4/{name}/{method}_b{b:g}_h{h}_s{seed}"
            out.append(f"[ -f {d}/{method}.json ] || {{ mkdir -p {d} && PYTHONPATH=. .venv-core/bin/python -u -m "
                       f"recal.cells --method {method} {'--strict-modes ' if method == 'recal' else ''}--prep {prep} "
                       f"--stream {stream} --budget {b:g} "
                       f"{hist}--seed {seed} {extra} --out {d} "
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
                       f"recal.cells --method recal --strict-modes --prep {prep} --stream {stream} --budget {budget:g} {flags} "
                       f"--seed {seed} {extra} --out {d} > {d}/log.txt 2>&1; }}  # recal.cells")
    return out


def shift_lines(prep_dir: str, name: str, shifts=("abrupt", "gradual", "correlated"), budget: float = 0.05,
                seeds=(0, 1, 2, 3, 4), signals=(True, False), extra: str = "") -> list[str]:
    """Shift timelines, one converted prep per (shift, seed) (scripts/convert_streams.py shift), with and
    without the rebuild signals."""
    out = []
    for kind in shifts:
        for on in signals:
            tag = "signals" if on else "periodic"
            for seed in seeds:
                prep = f"{prep_dir}/{name}_{kind}_seed{seed}"
                d = f"results/q4/{name}/shift_{kind}_{tag}_s{seed}"
                flag = "" if on else "--no-rebuild-signals"
                out.append(f"[ -f {d}/recal.json ] || {{ mkdir -p {d} && PYTHONPATH=. .venv-core/bin/python -u -m "
                           f"recal.cells --method recal --strict-modes --prep {prep} --stream {kind} --budget {budget:g} "
                           f"--seed {seed} {flag} {extra} --out {d} > {d}/log.txt 2>&1; }}  # recal.cells")
    return out


def burst_lines(prep: str, stream: str, name: str, budgets=(0.05,), seeds=(0, 1), extra: str = "") -> list[str]:
    """Burst experiment: the stream at each budget with and without the mini cache (burst periods are marked
    from the records: a request is in a burst when its query occurs 5 times in its 1,000-request block)."""
    out = []
    for budget in budgets:
        for label, flag in (("mini", ""), ("nomini", "--no-mini-cache")):
            for seed in seeds:
                d = f"results/q4/{name}/burst_b{budget:g}_{label}_s{seed}"
                out.append(f"[ -f {d}/recal.json ] || {{ mkdir -p {d} && PYTHONPATH=. .venv-core/bin/python -u -m "
                           f"recal.cells --method recal --strict-modes --prep {prep} --stream {stream} --budget {budget:g} "
                           f"--seed {seed} {flag} {extra} --out {d} > {d}/log.txt 2>&1; }}  # recal.cells")
    return out


UPDATE_SYSTEMS = {
    "recal": ("recal", "recal", "", True),
    "aker": ("aker", "aker", "", False),
    "proximity-a": ("proximity", "proximity_p0.1", "--prox-percentile 0.1 --tag _p0.1", False),
    "proximity-b": ("proximity", "proximity_p1", "--prox-percentile 1 --tag _p1", False),
}


def update_job_id(name: str, mode: str, rate: float, system: str, seed: int) -> str:
    return f"{name}_{mode}{round(100 * rate):d}_{system}_s{seed}"


def update_lines(w: dict, name: str, q4: dict, seeds) -> list[str]:
    """Data-update cells, one independent job per (workload, mode, rate, system, seed).

    Job contract: each job reads its own prep folder (the schedule and its Aker-format files, written by
    scripts/convert_streams.py updates --out) and, for ReCal (which searches a live database),
    the DSN of its own PostgreSQL clone in $PG_CLONES/<job id>.dsn. Nothing is shared or mutated between jobs;
    results go to the job's own directory.
    """
    out = []
    for mode in q4["update_modes"]:
        for rate in q4["update_rates"]:
            prep = f"{q4['update_prep_dir']}/{name}_{mode}{round(100 * rate):d}"
            for system in q4["update_systems"]:
                method, stem, flags, live = UPDATE_SYSTEMS[system]
                for seed in seeds:
                    job = update_job_id(name, mode, rate, system, seed)
                    d = f"results/q4/{name}/updates/{job}"
                    backend = (f"--backend pgvector --pg-dsn @$PG_CLONES/{job}.dsn --updates {prep}/updates.npz"
                               if live else f"--backend exact --updates {prep}/updates.npz")
                    out.append(f"[ -f {d}/{stem}.json ] || {{ mkdir -p {d} && PYTHONPATH=. .venv-core/bin/python -u "
                               f"-m recal.cells --method {method} {'--strict-modes ' if method == 'recal' else ''}--prep {prep} "
                               f"--stream {w['stream']} "
                               f"--budget {w['q2_budget']:g} --coord-bytes {w['coord_bytes']:g} {backend} {flags} "
                               f"--seed {seed} --out {d} > {d}/log.txt 2>&1; }}  # recal.cells")
    return out
