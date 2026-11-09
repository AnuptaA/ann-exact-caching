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
TP=${TP:-$HOME/third_party}
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

g++ -O2 -std=c++17 -shared -fPIC -DBOOST_LOG_DYN_LINK -I"$TP/aker/inc" -I"$TP/boost-install/include" \
  -I"$TP/faiss-install/include" "$HERE/aker_shim.cc" \
  -L"$TP/aker/build/lib" -laker -L"$TP/boost-install/lib" -lboost_log -lboost_thread \
  -Wl,-rpath,"$TP/aker/build/lib" -Wl,-rpath,"$TP/boost-install/lib" -o "$HERE/libakershim.so"

echo "built $HERE/libakershim.so"
echo "export LD_LIBRARY_PATH=$TP/aker/build/lib:$TP/boost-install/lib:$TP/faiss-install/lib64:\$LD_LIBRARY_PATH"
