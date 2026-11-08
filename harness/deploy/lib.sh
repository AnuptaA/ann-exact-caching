#!/usr/bin/env bash
# Shared helpers for the deploy scripts (sourced, not executed).

log() { printf '[q1 %s] %s\n' "$(date +%T)" "$*"; }
# die: Q1_ERR_FILE, when set by the setup step runner, keeps the message for the summary.
die() {
    printf '[q1] ERROR: %s\n' "$*" >&2
    [ -z "${Q1_ERR_FILE:-}" ] || printf '%s\n' "$*" > "$Q1_ERR_FILE"
    exit 1
}

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

# first dotted version number in a string ("cmake version 3.22.1" -> 3.22.1); empty, status 0, when none.
first_version() {
    printf '%s\n' "$1" | grep -oE '[0-9]+(\.[0-9]+)+' | head -n1 || true
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

# num_ge A B / num_add A B: decimal arithmetic for poll counters (poll intervals may be fractional).
num_ge() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a >= b) }'; }
num_add() { awk -v a="$1" -v b="$2" 'BEGIN { print a + b }'; }

APT_LOCKS_DEFAULT="/var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/lib/apt/lists/lock /var/cache/apt/archives/lock"

# pid_command PID: full command line of a process.
pid_command() {
    local pid="$1" cmd=""
    [ ! -r "/proc/$pid/cmdline" ] || cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    [ -n "$cmd" ] || cmd=$(ps -o args= -p "$pid" 2>/dev/null || true)
    printf '%s\n' "${cmd% }"
}

# apt_lock_holders: one "<lock file> <pid>" line per process holding an apt/dpkg lock. Reads the kernel lock
# table (/proc/locks) by inode, so it needs no root and never decides by process name.
apt_lock_holders() {
    local table="${Q1_PROC_LOCKS:-/proc/locks}" lock inode
    local -a locks
    read -ra locks <<< "${Q1_APT_LOCKS:-$APT_LOCKS_DEFAULT}"
    for lock in "${locks[@]}"; do
        [ -e "$lock" ] || continue
        inode=$(stat -c %i "$lock" 2>/dev/null || stat -f %i "$lock" 2>/dev/null || true)
        [ -n "$inode" ] || continue
        awk -v ino="$inode" -v lock="$lock" \
            '$2 != "->" { n = split($6, a, ":"); if (n == 3 && a[3] == ino) print lock, $5 }' "$table" | sort -u
    done
}

# describe_holders "<lock> <pid>" lines: one readable line per holder.
describe_holders() {
    local lock pid
    while read -r lock pid; do
        printf '  %s held by pid %s: %s\n' "$lock" "$pid" "$(pid_command "$pid")"
    done <<< "$1"
}

# wait_for_apt: wait until no process holds an apt/dpkg lock, at most Q1_APT_WAIT_SECONDS (default 900).
# Fails with the holder's pid and command line. Never kills a process and never touches a lock file.
wait_for_apt() {
    local table="${Q1_PROC_LOCKS:-/proc/locks}"
    if [ ! -r "$table" ]; then
        log "WARNING: cannot read $table; relying on apt's own lock timeout"
        return 0
    fi
    local limit="${Q1_APT_WAIT_SECONDS:-900}" poll="${Q1_APT_POLL_SECONDS:-5}"
    local waited=0 next_log=0 holders
    while :; do
        holders=$(apt_lock_holders)
        [ -n "$holders" ] || return 0
        if num_ge "$waited" "$limit"; then
            die "apt/dpkg locks still held after ${limit}s:"$'\n'"$(describe_holders "$holders")"
        fi
        if num_ge "$waited" "$next_log"; then
            log "waiting for apt/dpkg locks (${waited}s of ${limit}s):"
            describe_holders "$holders"
            next_log=$((next_log + 60))
        fi
        sleep "$poll"
        waited=$(num_add "$waited" "$poll")
    done
}

# wait_for_cloud_init: first-boot cloud-init runs apt; wait for it by its own status, at most
# Q1_CLOUD_INIT_WAIT_SECONDS (default 600).
wait_for_cloud_init() {
    command -v cloud-init >/dev/null 2>&1 || return 0
    local limit="${Q1_CLOUD_INIT_WAIT_SECONDS:-600}" poll="${Q1_APT_POLL_SECONDS:-5}" waited=0 state
    while :; do
        state=$(cloud-init status 2>/dev/null | awk '/^status:/ { print $2; exit }' || true)
        if [ "$state" != "running" ]; then
            log "cloud-init status: ${state:-unknown}"
            return 0
        fi
        num_ge "$waited" "$limit" && die "cloud-init still running after ${limit}s"
        log "waiting for cloud-init to finish (${waited}s of ${limit}s)"
        sleep "$poll"
        waited=$(num_add "$waited" "$poll")
    done
}

# require_sudo: fail at once instead of hanging on a password prompt.
require_sudo() {
    sudo -n true 2>/dev/null || die "passwordless sudo is required (sudo -n true failed)"
}

# apt_get <args>: non-interactive, time-bounded apt (no debconf, no needrestart prompts, waits for the locks).
apt_get() {
    [ -n "${Q1_DRYRUN:-}" ] || wait_for_apt
    run sudo env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a NEEDRESTART_SUSPEND=1 \
        apt-get -y -o DPkg::Lock::Timeout=900 -o Acquire::Retries=3 -o Acquire::http::Timeout=60 \
        -o Acquire::https::Timeout=60 -o Dpkg::Options::=--force-confold "$@"
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
