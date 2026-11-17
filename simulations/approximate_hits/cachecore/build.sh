#!/bin/sh
# Builds the cache core extension (_cachecore*.so, next to this script) for .venv-core (Python 3.11).
#   sh simulations/approximate_hits/cachecore/build.sh
set -e
cd "$(dirname "$0")"
PY=${PY:-../../../.venv-core/bin/python}
EXT=$($PY -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
INC=$($PY -m pybind11 --includes)
c++ -O3 -march=native -DNDEBUG -std=c++17 -shared -fPIC -pthread -fopenmp-simd -Wall -Wno-sign-compare $INC \
    cachecore.cpp -o _cachecore$EXT
echo "built $(pwd)/_cachecore$EXT"
