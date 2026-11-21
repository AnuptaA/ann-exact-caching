"""Aker run configuration and the budget rule."""

from dataclasses import asdict, dataclass

BUDGET_PERCENTS = (1, 5, 10)
K = 10

# Paper values (Sec. 4 and 5): alpha_T 0.25, alpha_L 1.1, query filter m 4,
# efConstruction 8, efSearch 16, top_delta 0.
PAPER = {"alpha_tighten": 0.25, "alpha_loosen": 1.1, "filter_m": 4,
         "filter_ef_construction": 8, "filter_ef_search": 16, "top_delta": 0}


def pool_size_for(n_vectors: int, percent: int) -> int:
    """Pool size, in pooled vectors, for a budget of exactly 1%, 5% or 10% of N.

    Rounds half up; fails on any other percentage.
    """
    if percent not in BUDGET_PERCENTS:
        raise ValueError(f"budget must be exactly one of {BUDGET_PERCENTS} percent, got {percent}")
    pool = (n_vectors * percent + 50) // 100
    if abs(pool * 100 - n_vectors * percent) > 50:
        raise ValueError("pool size rounding error exceeds one vector")
    return pool


@dataclass(frozen=True)
class AkerSettings:
    """One Aker or Proximity configuration (everything the driver needs)."""
    mode: str                  # "standard" or "proximity"
    pool_size: int
    alpha_tighten: float = PAPER["alpha_tighten"]
    alpha_loosen: float = PAPER["alpha_loosen"]
    global_thresh: float = 0.0  # squared L2; Proximity only
    filter_m: int = PAPER["filter_m"]
    filter_ef_construction: int = PAPER["filter_ef_construction"]
    filter_ef_search: int = PAPER["filter_ef_search"]
    top_delta: int = PAPER["top_delta"]
    k: int = K

    def __post_init__(self):
        if self.mode not in ("standard", "proximity"):
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.mode == "proximity" and not self.global_thresh > 0:
            raise ValueError("proximity needs a positive global_thresh (squared L2)")
        if self.mode == "standard" and self.global_thresh != 0:
            raise ValueError("standard mode does not use global_thresh")
        if self.pool_size < self.k:
            raise ValueError(f"pool_size {self.pool_size} is smaller than k={self.k}")

    def driver_args(self) -> list[str]:
        args = {"pool-size": self.pool_size, "k": self.k, "top-delta": self.top_delta,
                "alpha-tighten": self.alpha_tighten, "alpha-loosen": self.alpha_loosen,
                "filter-m": self.filter_m, "filter-ef-construction": self.filter_ef_construction,
                "filter-ef-search": self.filter_ef_search}
        if self.mode == "proximity":
            args["global-thresh"] = self.global_thresh
        return [x for key, value in args.items() for x in (f"--{key}", str(value))]

    def as_dict(self) -> dict:
        return asdict(self)
