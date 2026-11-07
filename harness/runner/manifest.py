"""The experiment grid: dataset x stream x system x setting x budget (one stream seed first)."""

import hashlib
import json
from dataclasses import asdict, dataclass

from harness.config import BUDGET_PERCENTS

ZIPF_STREAMS = ("zipf_0.30", "zipf_0.60", "zipf_0.99")
PROXIMITY_PERCENTILES = (0.1, 1.0)


@dataclass(frozen=True)
class Cell:
    dataset: str
    stream: str
    system: str      # "aker" or "proximity"
    setting: str     # "paper" for aker, "p0.1" / "p1" for proximity percentiles
    budget_percent: int
    stream_seed: int = 0

    @property
    def cell_id(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        name = f"{self.dataset}-{self.stream}-{self.system}-{self.setting}-b{self.budget_percent}-s{self.stream_seed}"
        return f"{name}-{hashlib.sha1(blob).hexdigest()[:8]}"

    def percentile(self) -> float:
        if self.system != "proximity":
            raise ValueError("only proximity cells have a percentile")
        return float(self.setting[1:])


def default_streams(dataset: str) -> tuple[str, ...]:
    streams = ZIPF_STREAMS + ("uniform",)
    return streams + ("no_locality",) if dataset in ("esci", "sphere") else streams


def build_grid(datasets, streams=None, budgets=BUDGET_PERCENTS, seeds=(0,),
               percentiles=PROXIMITY_PERCENTILES) -> list[Cell]:
    """Every cell of the grid, sorted by id so shards are stable."""
    cells = []
    for dataset in datasets:
        for stream in (streams or default_streams(dataset)):
            for seed in seeds:
                for budget in budgets:
                    cells.append(Cell(dataset, stream, "aker", "paper", budget, seed))
                    cells += [Cell(dataset, stream, "proximity", f"p{p:g}", budget, seed)
                              for p in percentiles]
    return sorted(set(cells), key=lambda c: c.cell_id)


def shard(cells: list[Cell], index: int, count: int) -> list[Cell]:
    if not 0 <= index < count:
        raise ValueError(f"shard {index} of {count}")
    return [c for i, c in enumerate(cells) if i % count == index]
