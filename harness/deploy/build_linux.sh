#!/usr/bin/env bash
# Build both Aker replay drivers on a Linux x86_64 instance (Release).
# Inputs (set by setup_instance.sh, or by hand): CMAKE_BIN, BOOST_ROOT (empty = system Boost), FAISS_ROOT.
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck source-path=SCRIPTDIR source=lib.sh
# shellcheck source-path=SCRIPTDIR source=lib.sh
. "$root/harness/deploy/lib.sh"
# shellcheck source-path=SCRIPTDIR source=pins.env
# shellcheck source-path=SCRIPTDIR source=pins.env
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

# A static libfaiss.a needs the BLAS and LAPACK it was built with on the link line.
faiss_static=0
if [ -n "${FAISS_ROOT:-}" ] && compgen -G "$FAISS_ROOT/lib*/libfaiss.a" > /dev/null; then
    faiss_static=1
    if [ "$(uname -s)" = Linux ]; then
        blas_listed=$("$(command -v ldconfig || echo /sbin/ldconfig)" -p 2>/dev/null | grep -c 'libopenblas\.so' || true)
        [ "${blas_listed:-0}" -gt 0 ] || die "libfaiss.a needs BLAS and LAPACK but no libopenblas was found: install libopenblas-dev"
    fi
    if [ -f "$FAISS_ROOT/BLAS_INFO" ]; then
        blas_vendor=$(sed -n 's/^BLA_VENDOR=//p' "$FAISS_ROOT/BLAS_INFO" | head -n1)
        [ -z "$blas_vendor" ] || args+=(-DBLA_VENDOR="$blas_vendor")
    fi
fi

"$cmake_bin" -S "$root/harness/cpp" -B "$build" "${args[@]}"
if [ "$faiss_static" -eq 1 ] && [ -f "$FAISS_ROOT/BLAS_INFO" ]; then
    check_blas_match "$FAISS_ROOT/BLAS_INFO" "$build/blas_info.txt"
fi
"$cmake_bin" --build "$build" -j "$(nproc)"
if [ "$(uname -s)" = Linux ] && command -v ldd > /dev/null 2>&1; then
    for exe in ak_replay_standard ak_replay_proximity; do
        unresolved=$(ldd "$build/$exe" | grep 'not found' || true)
        [ -z "$unresolved" ] || die "$exe has unresolved libraries: $unresolved"
    done
fi
{
    echo "commit $(git -C "$root" rev-parse HEAD)"
    echo "cmake $cmake_ver"
    echo "gcc $(first_version "$(gcc --version | head -n1)")"
    echo "boost ${BOOST_ROOT:-system}"
    echo "faiss ${FAISS_ROOT:-system}"
    toolchain_blas_lines "$build"
} > "$build/TOOLCHAIN.txt"
git -C "$root" rev-parse HEAD > "$build/BUILT_FROM_COMMIT"
log "built ak_replay_standard and ak_replay_proximity in $build"
