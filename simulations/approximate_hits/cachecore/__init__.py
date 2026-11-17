"""Cache core (doc 9ah): the request path of our cache in C++ (cachecore.cpp), shared by the headline and
timing runs. Build with `sh simulations/approximate_hits/cachecore/build.sh` (Python 3.11, .venv-core)."""
from ._cachecore import Core, LookupResult, QVCore  # noqa: F401

FEATURES = 7
N_BALLS = 3
POOR_FIT = 10.0
