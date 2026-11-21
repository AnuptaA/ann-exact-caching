#!/bin/sh
# Build the real Aker (github.com/sjoon-oh/aker; caching logic unmodified, one
# memory bug fixed, see below) and our shim.
# Everything lives outside the repo, in $TP (default ~/third_party):
#   $TP/aker             the authors' repository (git clone --depth 1; commit 6bb326a)
#   $TP/aker/extern/xxHash/{xxhash.h,xxh3.h}   xxHash v0.8.2 headers
#   $TP/aker/extern/YCSB-C                      git clone of basicthinker/YCSB-C
#   $TP/boost-install    Boost 1.86 (program_options, log, date_time, thread, system,
#                        filesystem, regex, atomic, chrono; b2, shared, release)
#   $TP/faiss-install    FAISS 1.8.0 C++ (cmake, shared, avx2, no GPU / Python)
# Aker is built in Standard mode (the adaptive per-entry threshold).
# Usage: sh build.sh   (then run with the LD_LIBRARY_PATH printed at the end)
set -e
TP=${TP:-${RECAL_THIRD_PARTY:-$HOME/third_party}}
HERE=$(cd "$(dirname "$0")" && pwd)

# One bug fix (aker_stackbuffer_fix.patch): StackFloatBuffer called alloca() in its
# constructor, so its buffer was dead stack memory once the constructor returned;
# the approximate filter then indexed / searched garbage and found no candidates
# (zero approximate hits). The fix only changes where that scratch buffer lives.
if git -C "$TP/aker" apply --check "$HERE/aker_stackbuffer_fix.patch" 2>/dev/null; then
  git -C "$TP/aker" apply "$HERE/aker_stackbuffer_fix.patch"
fi

cmake -S "$TP/aker" -B "$TP/aker/build" -DCMAKE_BUILD_TYPE=Release \
  -DBOOST_ROOT="$TP/boost-install" -DCMAKE_PREFIX_PATH="$TP/boost-install" \
  -DFAISS_ROOT="$TP/faiss-install" -DAKER_ENABLE_PROXIMITY_MODE=OFF -DAKER_ENABLE_POTLUCK_MODE=OFF
cmake --build "$TP/aker/build" -j 32 --target aker

shim() {   # $1 = Aker build dir, $2 = output .so (written to a temp file, then renamed: running jobs
           # keep their mapped copy)
  g++ -O2 -std=c++17 -shared -fPIC -DBOOST_LOG_DYN_LINK -I"$TP/aker/inc" -I"$TP/boost-install/include" \
    -I"$TP/faiss-install/include" "$HERE/aker_shim.cc" \
    -L"$1/lib" -laker -L"$TP/boost-install/lib" -lboost_log -lboost_thread \
    -Wl,-rpath,"$1/lib" -Wl,-rpath,"$TP/boost-install/lib" -o "$2.tmp" && mv "$2.tmp" "$2"
}
shim "$TP/aker/build" "$HERE/libakershim.so"

# Proximity mode (the authors' compile option: one fixed global threshold; the paper's
# "Proximity+" baseline), for tuning Aker to reach a recall target (doc 9j). Aker's CMakeLists
# hard-codes its output to <source>/build/lib, so a second build dir in the same tree would
# OVERWRITE the Standard library: build from a separate copy of the sources, $TP/aker_prox.
# Jobs using libakershim_prox.so must put $TP/aker_prox/build/lib FIRST in LD_LIBRARY_PATH
# (it takes precedence over the shim's compiled-in run path).
rsync -a --exclude build --exclude .git "$TP/aker/" "$TP/aker_prox/"
cmake -S "$TP/aker_prox" -B "$TP/aker_prox/build" -DCMAKE_BUILD_TYPE=Release \
  -DBOOST_ROOT="$TP/boost-install" -DCMAKE_PREFIX_PATH="$TP/boost-install" \
  -DFAISS_ROOT="$TP/faiss-install" -DAKER_ENABLE_PROXIMITY_MODE=ON -DAKER_ENABLE_POTLUCK_MODE=OFF
cmake --build "$TP/aker_prox/build" -j 32 --target aker
shim "$TP/aker_prox/build" "$HERE/libakershim_prox.so"
echo "built $HERE/libakershim.so and $HERE/libakershim_prox.so"
echo "export LD_LIBRARY_PATH=$TP/aker/build/lib:$TP/boost-install/lib:$TP/faiss-install/lib64:\$LD_LIBRARY_PATH"
