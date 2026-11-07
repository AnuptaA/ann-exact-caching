#!/usr/bin/env bash
# Build both Aker replay drivers on a Linux x86_64 instance (Release).
# Inputs (set by setup_instance.sh, or by hand): CMAKE_BIN, BOOST_ROOT (empty = system Boost), FAISS_ROOT.
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck source=lib.sh
. "$root/harness/deploy/lib.sh"
# shellcheck source=pins.env
. "$root/harness/deploy/pins.env"
build="${BUILD_DIR:-$root/build/replay}"
cmake_bin="${CMAKE_BIN:-cmake}"

if [ -z "${Q1_SETUP_ALLOW_NON_LINUX:-}" ] && { [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; }; then
    die "expected Linux x86_64, found $(uname -s) $(uname -m)"
fi
if [ -n "$(git -C "$root" status --porcelain)" ]; then
    die "git tree is dirty; commit or stash first"
fi
need_cmd "$cmake_bin"
cmake_ver=$(first_version "$("$cmake_bin" --version | head -n1)")
version_ge "${cmake_ver:-0}" 3.10 || die "cmake $cmake_ver is too old"

args=(-DCMAKE_BUILD_TYPE=Release)
[ -z "${FAISS_ROOT:-}" ] || args+=(-DFAISS_ROOT="$FAISS_ROOT")
if [ -n "${BOOST_ROOT:-}" ]; then
    args+=(-DBOOST_ROOT="$BOOST_ROOT" -DBoost_NO_SYSTEM_PATHS=ON)
fi

"$cmake_bin" -S "$root/harness/cpp" -B "$build" "${args[@]}"
"$cmake_bin" --build "$build" -j "$(nproc)"
{
    echo "commit $(git -C "$root" rev-parse HEAD)"
    echo "cmake $cmake_ver"
    echo "gcc $(first_version "$(gcc --version | head -n1)")"
    echo "boost ${BOOST_ROOT:-system}"
    echo "faiss ${FAISS_ROOT:-system}"
} > "$build/TOOLCHAIN.txt"
git -C "$root" rev-parse HEAD > "$build/BUILT_FROM_COMMIT"
log "built ak_replay_standard and ak_replay_proximity in $build"
