"""A job of the experiment matrix: its shell line and what the shard runner needs to know about it."""

from dataclasses import dataclass

# stage order inside a shard: hit-rate and recall runs first, then the split sweep (needs their records), the choice
# of each result cache's best split, and last the thread lines (need the best splits)
STAGES = {"hit": 0, "update": 0, "replay": 1, "best_split": 2, "threads": 3}
EXCLUSIVE_KINDS = ("replay", "threads")


@dataclass(frozen=True)
class Job:
    line: str
    exp: str
    workload: str | None
    system: str
    kind: str
    cell: str
    needs: tuple = ()
    outputs: tuple = ()
    inputs: tuple = ()

    @property
    def produced(self) -> tuple:
        """Output paths: a directory ends with a slash. A run's own directory unless the job lists files."""
        return self.outputs or (self.cell.rstrip("/") + "/",)

    @property
    def stage(self) -> int:
        return STAGES[self.kind]

    @property
    def exclusive(self) -> bool:
        """Throughput runs need a quiet machine: they run alone, one after the other."""
        return self.kind in EXCLUSIVE_KINDS
