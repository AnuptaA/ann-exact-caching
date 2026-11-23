"""Figure inputs derived from per-request records only (recal/records.py)."""

from recal.analysis.composition import hit_composition
from recal.analysis.guarantee import guarantee_table
from recal.analysis.interp import hit_rate_at_recall
from recal.analysis.oracle import oracle
from recal.analysis.rates import rates

__all__ = ["guarantee_table", "hit_composition", "hit_rate_at_recall", "oracle", "rates"]
