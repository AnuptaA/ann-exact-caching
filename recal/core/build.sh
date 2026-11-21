#!/bin/sh
# Builds the ReCal core extension (_recal_core*.so, next to this script) for .venv-core (Python 3.11).
#   sh recal/core/build.sh
# Reuses the vendored hnswlib headers of the advisor's cache core (not modified).
set -e
cd "$(dirname "$0")"
PY=${PY:-../../.venv-core/bin/python}
# CORE_ARCH: use a portable value such as -march=x86-64-v3 for binaries moved between machines
CORE_ARCH=${CORE_ARCH:--march=native}
EXT=$($PY -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
INC=$($PY -m pybind11 --includes)
HNSW=../../simulations/approximate_hits/cachecore/third_party
# macOS links Python extension modules with unresolved Python symbols
LDX=""
[ "$(uname)" = "Darwin" ] && LDX="-undefined dynamic_lookup"
c++ -O3 $CORE_ARCH -DNDEBUG -std=c++17 -shared -fPIC -pthread -Wall -Wno-sign-compare $INC -I$HNSW $LDX \
    recal_core.cpp -o _recal_core$EXT.tmp
mv -f _recal_core$EXT.tmp _recal_core$EXT   # atomic: running jobs keep the old file
echo "built $(pwd)/_recal_core$EXT"
