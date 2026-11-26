"""E9: ReCal (v10's greedy construction) against the LRU ablation, same budget, idealized mode, whole stream.

The stream's first part is history and the rest is scored (recal.sim). Four runs per budget: greedy and LRU, each
with the mini cache on and off.
"""

import tempfile
from pathlib import Path

from recal import sim
from recal.analysis.guarantee import guarantee_table
from recal.analysis.rates import rates as rates_of
from recal.records import Records

NAME = "E9"
VARIANTS = (("recal", False), ("lru", True))
WINDOW = 5000


def sim_args(prep: str, stream: str, budget: float, coord_bytes: float, lru: bool, mini: bool, out: str,
             extra: list[str] | None = None):
    argv = ["--prep", prep, "--stream", stream, "--budget", str(budget), "--coord-bytes", str(coord_bytes),
            "--backend", "stored", "--idealized", "--out", out, *(extra or [])]
    if lru:
        argv.append("--lru")
    if not mini:
        argv.append("--no-mini-cache")
    return sim.parser().parse_args(argv)


def one_run(prep, stream, budget, coord_bytes, lru, mini, extra=None, window=WINDOW) -> dict:
    with tempfile.TemporaryDirectory(prefix="micro_e9_") as out:
        records = Records()
        result = sim.run(sim_args(prep, stream, budget, coord_bytes, lru, mini, out, extra), records)
    cols = records.arrays()
    r = rates_of(cols)
    g = guarantee_table(cols, 0.9, window)
    return dict(hit_rate=r["hit_rate"], recall_of_hits=r["recall_of_hits"], mini_hit_rate=r["mini_hit_rate"],
                main_hit_rate=r["main_hit_rate"],
                mini_share_of_hits=r["mini_hit_rate"] / r["hit_rate"] if r["hit_rate"] else float("nan"),
                share_windows_below_target=g["share_below_target"], n_windows=g["n_windows"],
                n_scored=r["n"], constructions=result["constructions"])


def variants() -> list[tuple[str, bool, bool]]:
    """(method, lru, mini cache) of the four runs per budget, in row order."""
    return [(method, lru, mini) for method, lru in VARIANTS for mini in (True, False)]


def run_variant(workload: str, prep: str, stream: str, budget: float, coord_bytes: float, method: str, lru: bool,
                mini: bool, extra: list[str] | None = None, window: int = WINDOW) -> dict:
    return dict(workload=workload, budget=budget, method=method, mini_cache=mini,
                **one_run(prep, stream, budget, coord_bytes, lru, mini, extra, window))


def run(workload: str, prep: str, stream: str, budget: float, coord_bytes: float, extra: list[str] | None = None,
        window: int = WINDOW) -> list[dict]:
    return [run_variant(workload, prep, stream, budget, coord_bytes, method, lru, mini, extra, window)
            for method, lru, mini in variants()]


def sim_extra(params: dict) -> list[str]:
    """Simulator flags for the Params overrides the micro-experiments accept (tests shrink these)."""
    flags = {"H": "--history", "cold_start": "--cold-start", "min_ess": "--min-ess"}
    return [x for key, flag in flags.items() if key in params for x in (flag, str(params[key]))]
