"""ReCal parameters (spec section P). Defaults are the spec's."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    K: int = 10                     # neighbors per request (nominal, used for sizing)
    k_max: int = 100                # largest K the cache answers (Q1 step 1)
    T: float = 0.9                  # recall target
    delta: float = 0.05             # 1 - delta is the confidence
    a: float = 0.1                  # audit rate
    H: int = 30_000                 # history window, main-path requests
    cold_start: int = 10_000        # requests before the first construction
    roles: tuple = (0.2, 0.4, 0.2, 0.2)  # candidate, selection, training, calibration
    sizes: tuple = (1, 2, 3, 4, 5)  # entry sizes, multiples of K
    shell_width: int = 2            # shell ranks jK+1 .. (j+2)K
    deep: int = 7                   # deepest fetch, multiples of K
    bits: int = 4                   # shell quantization
    m_percentile: float = 95.0      # entries probed per query
    n_thresholds: int = 100         # quantiles of the training scores
    min_ess: float = 100.0
    window_items: float = 200.0     # W = min(H, window_items / (a h0))
    mini_percentile: float = 95.0
    mini_cap: float = 0.10          # of M
    mini_floor: int = 256           # entries
    headroom: float = 0.02          # of M
    sweep: float = 0.05             # of the pool
    insert_batch_min: int = 100
    insert_batch_frac: float = 0.001
    insert_rounds: int = 5
    insert_centers: int = 16
    shrink: float = 0.8
    frozen: float = 0.05
    min_interval: float = 0.2       # of H
    drop_window: float = 0.2        # of H
    drop_floor: int = 2000
    drop_every: int = 1000
    drop_from: float = 0.4          # of H
    drop_se: float = 3.0
    outage: float = 0.9
    split_by_query: bool = False    # B5 option: all occurrences of a query in one role
    idealized: bool = False         # exact labels and candidates from all entries (no m cap)
    seed: int = 0
    item_bytes: int = 8             # window item: bin, recall count, weight code, request index

    @property
    def window_cap(self) -> int:
        return self.H
