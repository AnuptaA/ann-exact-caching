#!/usr/bin/env bash
# Build both Aker replay drivers on a Linux x86_64 instance (Release).
# Needs cmake, a C++17 compiler, libfaiss-dev (or FAISS_ROOT), libboost >= 1.80 and OpenMP.
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
build="${BUILD_DIR:-$root/build/replay}"

if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
    echo "build_linux.sh: expected Linux x86_64, found $(uname -s) $(uname -m)" >&2
    exit 1
fi
if [ -n "$(git -C "$root" status --porcelain)" ]; then
    echo "build_linux.sh: git tree is dirty; commit or stash first" >&2
    exit 1
fi

cmake -S "$root/harness/cpp" -B "$build" -DCMAKE_BUILD_TYPE=Release \
    ${FAISS_ROOT:+-DFAISS_ROOT="$FAISS_ROOT"} ${BOOST_ROOT:+-DBOOST_ROOT="$BOOST_ROOT"}
cmake --build "$build" -j "$(nproc)"
git -C "$root" rev-parse HEAD > "$build/BUILT_FROM_COMMIT"
echo "built ak_replay_standard and ak_replay_proximity in $build"
