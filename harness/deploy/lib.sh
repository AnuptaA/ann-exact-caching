#!/usr/bin/env bash
# Shared helpers for the deploy scripts (sourced, not executed).

log() { printf '[q1 %s] %s\n' "$(date +%T)" "$*"; }
die() { printf '[q1] ERROR: %s\n' "$*" >&2; exit 1; }

# run <cmd...>: execute, or only print when Q1_DRYRUN is set (used by the decision-logic tests).
run() {
    if [ -n "${Q1_DRYRUN:-}" ]; then
        printf 'DRYRUN: %s\n' "$*"
        return 0
    fi
    "$@"
}

# version_ge A B: true when version A >= version B (dotted numbers, sort -V).
version_ge() {
    [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]
}

# first dotted version number in a string ("cmake version 3.22.1" -> 3.22.1).
first_version() {
    printf '%s\n' "$1" | grep -oE '[0-9]+(\.[0-9]+)+' | head -n1
}

# guard_path P: refuse paths that must never be deleted or recreated (empty, relative, short, system roots).
guard_path() {
    local p="${1:-}"
    [ -n "$p" ] || die "refusing an empty path"
    case "$p" in
        /*) ;;
        *) die "refusing a relative path: $p" ;;
    esac
    case "$p" in
        *..*) die "refusing a path with '..': $p" ;;
    esac
    local depth
    depth=$(printf '%s' "$p" | tr -cd '/' | wc -c)
    [ "$depth" -ge 2 ] || die "refusing a path this close to the root: $p"
    [ "${#p}" -ge 8 ] || die "refusing a short path: $p"
    case "$p" in
        /usr|/usr/*|/bin|/bin/*|/sbin|/sbin/*|/lib|/lib/*|/etc|/etc/*|/var|/var/lib|/boot|/root|/home|/opt|/data|/mnt|/tmp)
            die "refusing a system path: $p" ;;
    esac
}

# safe_rm_rf P: guard_path then remove.
safe_rm_rf() {
    guard_path "${1:-}"
    run rm -rf -- "$1"
}

# wait_for_apt: unattended upgrades hold the dpkg lock on fresh instances.
wait_for_apt() {
    local waited=0
    while pgrep -x unattended-upgr >/dev/null 2>&1 || pgrep -x apt-get >/dev/null 2>&1 || pgrep -x dpkg >/dev/null 2>&1; do
        [ "$waited" -lt 900 ] || die "apt/dpkg still busy after 15 minutes (unattended-upgrades?)"
        log "waiting for another apt/dpkg process to finish"
        sleep 10
        waited=$((waited + 10))
    done
}

# apt_get <args>: non-interactive apt (no debconf, no needrestart prompts, wait for the dpkg lock).
apt_get() {
    run sudo env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a NEEDRESTART_SUSPEND=1 \
        apt-get -y -o DPkg::Lock::Timeout=900 -o Dpkg::Options::=--force-confold "$@"
}

# need_cmd C: the command must exist.
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"; }

# sha256_check FILE EXPECTED
sha256_check() {
    local got
    if command -v sha256sum >/dev/null 2>&1; then
        got=$(sha256sum "$1" | cut -d' ' -f1)
    else
        got=$(shasum -a 256 "$1" | cut -d' ' -f1)
    fi
    [ "$got" = "$2" ] || die "sha256 mismatch for $1: got $got, expected $2"
}
