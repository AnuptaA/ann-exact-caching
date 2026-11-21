#!/bin/sh
# Build the real Aker (github.com/sjoon-oh/aker; caching logic unmodified, one
# memory bug fixed, see below) and our shim.
# Everything lives outside the repo, in $TP (default ~/third_party):
#   $TP/aker             the authors' repository (git clone --depth 1; commit 6bb326a)
#   $SRC/extern/xxHash/{xxhash.h,xxh3.h}   xxHash v0.8.2 headers
#   $SRC/extern/YCSB-C                      git clone of basicthinker/YCSB-C
#   $TP/boost-install    Boost 1.86 (program_options, log, date_time, thread, system,
#                        filesystem, regex, atomic, chrono; b2, shared, release)
#   $TP/faiss-install    FAISS 1.8.0 C++ (cmake, shared, avx2, no GPU / Python)
# Aker is built in Standard mode (the adaptive per-entry threshold).
# Usage: sh build.sh   (then run with the LD_LIBRARY_PATH printed at the end)
set -e
TP=${TP:-${RECAL_THIRD_PARTY:-$HOME/third_party}}
HERE=$(cd "$(dirname "$0")" && pwd)
# AKER_SRC=clone (default): the authors' clone at $TP/aker (commit 6bb326a), filter parameters hard-coded.
# AKER_SRC=vendored: the repository's third_party/aker (local patches in PATCHES.md), built from a copy at
# $TP/aker_vendored; it exposes the query filter's HNSW parameters and the shim sets the paper's values
# (m 4, efConstruction 8, efSearch 16; override with AKER_FILTER_M / AKER_FILTER_EFC / AKER_FILTER_EFS).
AKER_SRC=${AKER_SRC:-clone}
SHIM_DEFS=""
if [ "$AKER_SRC" = "vendored" ]; then
  SRC=$TP/aker_vendored
  mkdir -p "$SRC"
  rsync -a --exclude build --exclude .git "$HERE/../../../third_party/aker/" "$SRC/"
  SHIM_DEFS="-DRECAL_AKER_FILTER=1"
else
  SRC=$TP/aker
fi

# One bug fix (aker_stackbuffer_fix.patch): StackFloatBuffer called alloca() in its
# constructor, so its buffer was dead stack memory once the constructor returned;
# the approximate filter then indexed / searched garbage and found no candidates
# (zero approximate hits). The fix only changes where that scratch buffer lives.
if [ "$AKER_SRC" = "clone" ] && git -C "$SRC" apply --check "$HERE/aker_stackbuffer_fix.patch" 2>/dev/null; then
  git -C "$SRC" apply "$HERE/aker_stackbuffer_fix.patch"
fi

cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release \
  -DBOOST_ROOT="$TP/boost-install" -DCMAKE_PREFIX_PATH="$TP/boost-install" \
  -DFAISS_ROOT="$TP/faiss-install" -DAKER_ENABLE_PROXIMITY_MODE=OFF -DAKER_ENABLE_POTLUCK_MODE=OFF
cmake --build "$SRC/build" -j 32 --target aker

shim() {   # $1 = Aker build dir, $2 = output .so (written to a temp file, then renamed: running jobs
           # keep their mapped copy)
  g++ -O2 -std=c++17 -shared -fPIC -DBOOST_LOG_DYN_LINK $SHIM_DEFS -I"$SRC/inc" -I"$TP/boost-install/include" \
    -I"$TP/faiss-install/include" "$HERE/aker_shim.cc" \
    -L"$1/lib" -laker -L"$TP/boost-install/lib" -lboost_log -lboost_thread \
    -Wl,-rpath,"$1/lib" -Wl,-rpath,"$TP/boost-install/lib" -o "$2.tmp" && mv "$2.tmp" "$2"
}
shim "$SRC/build" "$HERE/libakershim.so"

# Proximity mode (the authors' compile option: one fixed global threshold; the paper's
# "Proximity+" baseline), for tuning Aker to reach a recall target (doc 9j). Aker's CMakeLists
# hard-codes its output to <source>/build/lib, so a second build dir in the same tree would
# OVERWRITE the Standard library: build from a separate copy of the sources, ${SRC}_prox.
# Jobs using libakershim_prox.so must put ${SRC}_prox/build/lib FIRST in LD_LIBRARY_PATH
# (it takes precedence over the shim's compiled-in run path).
rsync -a --exclude build --exclude .git "$SRC/" "${SRC}_prox/"
cmake -S "${SRC}_prox" -B "${SRC}_prox/build" -DCMAKE_BUILD_TYPE=Release \
  -DBOOST_ROOT="$TP/boost-install" -DCMAKE_PREFIX_PATH="$TP/boost-install" \
  -DFAISS_ROOT="$TP/faiss-install" -DAKER_ENABLE_PROXIMITY_MODE=ON -DAKER_ENABLE_POTLUCK_MODE=OFF
cmake --build "${SRC}_prox/build" -j 32 --target aker
shim "${SRC}_prox/build" "$HERE/libakershim_prox.so"
echo "built $HERE/libakershim.so and $HERE/libakershim_prox.so"
echo "export LD_LIBRARY_PATH=$SRC/build/lib:$TP/boost-install/lib:$TP/faiss-install/lib64:\$LD_LIBRARY_PATH"
