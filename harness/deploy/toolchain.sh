#!/usr/bin/env bash
# Toolchain for the Aker drivers and PostgreSQL. Sourced by setup_instance.sh after lib.sh and pins.env.
# Anything stock Ubuntu lacks or has too old is built from a pinned, verified source under $Q1_TOOLS.
# Variables set for the caller: CMAKE_BIN, BOOST_ROOT (empty when the system Boost is used), FAISS_ROOT.
# Expects from the caller: root, Q1_TOOLS, PYTHON_BIN.
# shellcheck disable=SC2154

ensure_build_packages() {
    apt_get update
    apt_get install build-essential git curl ca-certificates gnupg lsb-release pkg-config \
        python3-venv python3-dev bzip2 zstd libopenblas-dev libomp-dev libssl-dev
    local gcc_ver
    gcc_ver=$(first_version "$(gcc --version 2>/dev/null | head -n1)")
    [ -n "$gcc_ver" ] || gcc_ver=0
    version_ge "$gcc_ver" "$MIN_GCC" || die "gcc $gcc_ver found, need >= $MIN_GCC (C++17, OpenMP)"
}

# CMake >= MIN_CMAKE: use the system one if new enough, else a pinned pip wheel in a build venv.
ensure_cmake() {
    CMAKE_BIN=""
    if command -v cmake >/dev/null 2>&1; then
        local v
        v=$(first_version "$(cmake --version | head -n1)")
        if version_ge "${v:-0}" "$MIN_CMAKE"; then
            CMAKE_BIN=$(command -v cmake)
            log "cmake $v (system) is new enough"
            return 0
        fi
        log "system cmake ${v:-unknown} is older than $MIN_CMAKE; installing cmake==$CMAKE_PIP_VERSION in a build venv"
    else
        log "no system cmake; installing cmake==$CMAKE_PIP_VERSION in a build venv"
    fi
    run "$PYTHON_BIN" -m venv "$root/.venv-build"
    run "$root/.venv-build/bin/pip" install --quiet "cmake==$CMAKE_PIP_VERSION"
    CMAKE_BIN="$root/.venv-build/bin/cmake"
}

# Boost >= MIN_BOOST with the libraries the drivers link: system packages if new enough, else a pinned build.
ensure_boost() {
    BOOST_ROOT=""
    local sys
    sys=$(dpkg-query -W -f='${Version}' libboost-dev 2>/dev/null || true)
    if [ -z "$sys" ] && command -v apt-cache >/dev/null 2>&1; then
        sys=$(apt-cache policy libboost-dev 2>/dev/null | awk '/Candidate:/ {print $2}')
    fi
    sys=$(first_version "$sys")
    if [ -n "$sys" ] && version_ge "$sys" "$MIN_BOOST"; then
        log "system Boost $sys is new enough; installing the dev packages"
        apt_get install libboost-program-options-dev libboost-log-dev libboost-thread-dev \
            libboost-system-dev libboost-filesystem-dev libboost-regex-dev libboost-date-time-dev \
            libboost-atomic-dev libboost-chrono-dev
        return 0
    fi
    log "system Boost ${sys:-missing} is older than $MIN_BOOST; building Boost $BOOST_VERSION from a pinned tarball"
    BOOST_ROOT="$Q1_TOOLS/boost"
    local marker="$BOOST_ROOT/.installed-$BOOST_VERSION"
    if [ -f "$marker" ]; then
        log "Boost $BOOST_VERSION already built in $BOOST_ROOT"
        return 0
    fi
    local src="$Q1_TOOLS/src"
    guard_path "$src"
    run mkdir -p "$src"
    run curl -fL --retry 5 --retry-delay 10 -o "$src/boost.tar.bz2" "$BOOST_URL"
    [ -n "${Q1_DRYRUN:-}" ] || sha256_check "$src/boost.tar.bz2" "$BOOST_SHA256"
    safe_rm_rf "$src/boost_src"
    run mkdir -p "$src/boost_src"
    run tar -xjf "$src/boost.tar.bz2" -C "$src/boost_src" --strip-components=1
    run bash -c "cd '$src/boost_src' && ./bootstrap.sh --prefix='$BOOST_ROOT' --with-libraries='$BOOST_LIBS' && ./b2 -j\"\$(nproc)\" link=shared variant=release threading=multi install"
    run touch "$marker"
}

# FAISS pinned commit, CPU only, built with the chosen CMake into $Q1_TOOLS/faiss.
ensure_faiss() {
    FAISS_ROOT="$Q1_TOOLS/faiss"
    local marker="$FAISS_ROOT/.installed-$FAISS_COMMIT"
    if [ -f "$marker" ]; then
        log "FAISS $FAISS_VERSION already built in $FAISS_ROOT"
        return 0
    fi
    local src="$Q1_TOOLS/src/faiss"
    guard_path "$src"
    safe_rm_rf "$src"
    run mkdir -p "$src"
    run git -C "$src" init --quiet
    run git -C "$src" fetch --quiet --depth 1 https://github.com/facebookresearch/faiss "$FAISS_COMMIT"
    run git -C "$src" checkout --quiet FETCH_HEAD
    run "$CMAKE_BIN" -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_GPU=OFF \
        -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX="$FAISS_ROOT"
    run "$CMAKE_BIN" --build "$src/build" -j "$(nproc)"
    run "$CMAKE_BIN" --install "$src/build"
    run touch "$marker"
}

# PostgreSQL PG_MAJOR from PGDG (jammy ships 14; noble ships 16 but PGDG works for both).
ensure_postgres() {
    local pg_bin="/usr/lib/postgresql/$PG_MAJOR/bin"
    if [ ! -x "$pg_bin/pg_ctl" ]; then
        run sudo install -d /usr/share/postgresql-common/pgdg
        run sudo curl -fsSL -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc https://www.postgresql.org/media/keys/ACCC4CF8.asc
        run bash -c "echo \"deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt \$(lsb_release -cs)-pgdg main\" | sudo tee /etc/apt/sources.list.d/pgdg.list >/dev/null"
        apt_get update
        apt_get install "postgresql-$PG_MAJOR" "postgresql-server-dev-$PG_MAJOR" "postgresql-client-$PG_MAJOR"
    fi
    # our own cluster only: the packaged cluster must not run next to it
    run sudo systemctl disable --now postgresql || true
    run sudo pg_dropcluster --stop "$PG_MAJOR" main || true
    local ctl
    ctl="$("$pg_bin/pg_config" --sharedir 2>/dev/null || echo /nonexistent)/extension/vector.control"
    if [ -n "${Q1_DRYRUN:-}" ] || ! grep -q "default_version = '$PGVECTOR_VERSION'" "$ctl" 2>/dev/null; then
        local tmp
        tmp=$(mktemp -d)
        guard_path "$tmp"
        run git -C "$tmp" init --quiet
        run git -C "$tmp" fetch --quiet --depth 1 https://github.com/pgvector/pgvector "$PGVECTOR_COMMIT"
        run git -C "$tmp" checkout --quiet FETCH_HEAD
        run make -C "$tmp" PG_CONFIG="$pg_bin/pg_config"
        run sudo make -C "$tmp" PG_CONFIG="$pg_bin/pg_config" install
        safe_rm_rf "$tmp"
    fi
}
