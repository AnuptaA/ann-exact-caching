#!/bin/bash
# One-command setup of a freshly launched instance (Ubuntu 24.04, run as the normal user, here ubuntu):
#
#   curl -fsSL https://raw.githubusercontent.com/AnuptaA/ann-exact-caching/main/scripts/instance_setup.sh | bash -s -- <dataset>
#   bash scripts/instance_setup.sh [options] <dataset>
#
# <dataset> is one of spacev, triviaqa, medrag, tripclick, laion. Steps: system packages and the AWS CLI, the data
# volume at /data, the code at /data/thesis, the RECAL_* settings, secrets and agent context from the private
# bucket, Claude Code, and a tmux session "main" running the prepare bootstrap. Safe to rerun; stops at the first
# failure. Needs the instance role to read the bucket. Secret values are never printed.
#
# Options:
#   --dry-run          print every action without doing it
#   --bucket NAME      bucket of secrets, context and the store (default recal-eval-aa)
#   --prefix NAME      store prefix inside the bucket (default runs)
#   --repo-url URL     repository to clone (default https://github.com/AnuptaA/ann-exact-caching.git)
#   --no-claude        skip the Claude Code install
#   --no-tmux          skip the tmux session and the bootstrap
#
# Secrets (upload them with scripts/upload_secrets.sh), all under s3://<bucket>/ops/:
#   secrets/env.local      -> /data/thesis/.env.local (mode 600): the RECAL_DRIVE_* folder links
#   secrets/github_token   -> git credential store (~/.git-credentials, mode 600)
#   secrets/git_identity   -> git user.name (line 1) and user.email (line 2)
#   context/common.md and context/<dataset>.md -> ~/agent_context.md
set -eEu -o pipefail

BUCKET="recal-eval-aa"
PREFIX="runs"
REPO_URL="https://github.com/AnuptaA/ann-exact-caching.git"
DATA_DIR="/data"
REPO_DIR="/data/thesis"
DRY=0
WITH_CLAUDE=1
WITH_TMUX=1
DATASET=""
DONE=""
SKIPPED=""
MARK="recal settings"
DATASETS="spacev triviaqa medrag tripclick laion"

log() { echo "[instance_setup] $*"; }
die() { echo "[instance_setup] $*" >&2; exit 1; }
did() { DONE="$DONE  - $1"$'\n'; }
skipped() { SKIPPED="$SKIPPED  - $1"$'\n'; }

# Print the command in a dry run, run it otherwise.
run() {
    if [ "$DRY" -eq 1 ]; then echo "+ $*"; return 0; fi
    "$@"
}

on_error() {
    echo "[instance_setup] failed: '$2' (line $1). Fix the cause and run the same command again." >&2
}

# Pick the data volume from `lsblk -bnP -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,PKNAME` output on stdin: a disk with no
# filesystem, no mount and no partitions or children. The root disk always has a mounted partition, so it never
# qualifies. Prints /dev/<name>. Exit 0: one candidate; 1: none; 2: more than one (ambiguous).
pick_data_volume() {
    local result
    result=$(awk '
        function val(key,   pat) {
            pat = "(^| )" key "=\"[^\"]*\""
            if (match($0, pat)) {
                s = substr($0, RSTART, RLENGTH)
                sub("^ ", "", s)
                sub("^" key "=\"", "", s)
                sub("\"$", "", s)
                return s
            }
            return ""
        }
        {
            n++
            name[n] = val("NAME"); type[n] = val("TYPE"); fs[n] = val("FSTYPE"); mp[n] = val("MOUNTPOINT")
            parent = val("PKNAME")
            if (parent != "") has_child[parent] = 1
        }
        END {
            for (i = 1; i <= n; i++) {
                if (type[i] == "disk" && fs[i] == "" && mp[i] == "" && !(name[i] in has_child)) {
                    count++; pick = name[i]
                }
            }
            print count + 0, pick
        }')
    case "${result%% *}" in
        1) echo "/dev/${result#* }" ;;
        0) return 1 ;;
        *) return 2 ;;
    esac
}

usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; }

parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --dry-run) DRY=1 ;;
            --bucket) BUCKET="${2:?--bucket needs a name}"; shift ;;
            --prefix) PREFIX="${2:?--prefix needs a name}"; shift ;;
            --repo-url) REPO_URL="${2:?--repo-url needs a URL}"; shift ;;
            --no-claude) WITH_CLAUDE=0 ;;
            --no-tmux) WITH_TMUX=0 ;;
            -h | --help) usage; exit 0 ;;
            -*) die "unknown option: $1" ;;
            *) [ -z "$DATASET" ] || die "one dataset only (got $DATASET and $1)"; DATASET="$1" ;;
        esac
        shift
    done
    [ -n "$DATASET" ] || die "usage: instance_setup.sh [options] <dataset> (one of: $DATASETS)"
    case " $DATASETS " in *" $DATASET "*) ;; *) die "unknown dataset: $DATASET (one of: $DATASETS)" ;; esac
}

settings_block() {
    cat <<EOF
# >>> $MARK >>>
export RECAL_STORE=s3://$BUCKET/$PREFIX
export RECAL_REPO=$REPO_DIR
export RECAL_THIRD_PARTY=$DATA_DIR/third_party
export RECAL_QUEUE_DIR=$DATA_DIR/mx
export RECAL_TMP=$DATA_DIR/tmp
# <<< $MARK <<<
EOF
}

step_system() {
    log "step system"
    run sudo env DEBIAN_FRONTEND=noninteractive apt-get update -y
    run sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y git unzip curl tmux make ca-certificates
    if [ "$DRY" -eq 0 ] && command -v aws > /dev/null 2>&1; then
        skipped "aws cli: already installed"
    else
        local arch="x86_64"
        [ "$(uname -m)" = "aarch64" ] && arch="aarch64"
        run curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-$arch.zip" -o /tmp/awscliv2.zip
        run unzip -q -o /tmp/awscliv2.zip -d /tmp/awscli_install
        run sudo /tmp/awscli_install/aws/install --update
        did "aws cli v2 installed"
    fi
    did "system packages (git unzip curl tmux make)"
}

step_volume() {
    log "step data volume"
    if [ "$DRY" -eq 0 ] && mountpoint -q "$DATA_DIR"; then
        skipped "data volume: $DATA_DIR is already mounted"
        return 0
    fi
    if [ "$DRY" -eq 1 ]; then
        echo "+ find the one unformatted unpartitioned disk (lsblk), refuse if none or more than one"
        echo "+ sudo mkfs.ext4 -L recaldata <disk>   (skipped when a filesystem labelled recaldata exists)"
        echo "+ sudo mkdir -p $DATA_DIR; sudo mount LABEL=recaldata $DATA_DIR"
        echo "+ add 'LABEL=recaldata $DATA_DIR ext4 defaults,nofail 0 2' to /etc/fstab"
        echo "+ sudo chown $(id -un): $DATA_DIR"
        return 0
    fi
    local device="" code=0
    if sudo blkid -L recaldata > /dev/null 2>&1; then
        log "found an existing filesystem labelled recaldata; mounting it"
    else
        device=$(lsblk -bnP -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,PKNAME | pick_data_volume) || code=$?
        if [ "$code" -ne 0 ]; then
            lsblk
            [ "$code" -eq 1 ] && die "no unformatted data volume found; attach an EBS volume in the console and rerun"
            die "more than one unformatted volume; leave only the data volume attached (or format one with label recaldata)"
        fi
        log "formatting $device as ext4 (label recaldata)"
        sudo mkfs.ext4 -q -L recaldata "$device"
    fi
    sudo mkdir -p "$DATA_DIR"
    sudo mount LABEL=recaldata "$DATA_DIR"
    grep -q "LABEL=recaldata" /etc/fstab || echo "LABEL=recaldata $DATA_DIR ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab > /dev/null
    sudo chown "$(id -un):$(id -gn)" "$DATA_DIR"
    did "data volume mounted at $DATA_DIR${device:+ ($device formatted)}"
}

step_code() {
    log "step code"
    if [ "$DRY" -eq 0 ] && [ -d "$REPO_DIR/.git" ]; then
        run git -C "$REPO_DIR" pull --ff-only
        did "code fast-forwarded in $REPO_DIR"
    else
        run git clone "$REPO_URL" "$REPO_DIR"
        did "code cloned to $REPO_DIR"
    fi
    if [ "$DRY" -eq 1 ]; then
        echo "+ add datasets/aker/ and .env.local to $REPO_DIR/.git/info/exclude"
        return 0
    fi
    local exclude="$REPO_DIR/.git/info/exclude" entry
    mkdir -p "$(dirname "$exclude")"
    for entry in "datasets/aker/" ".env.local"; do
        grep -qxF "$entry" "$exclude" 2> /dev/null || echo "$entry" >> "$exclude"
    done
}

step_settings() {
    log "step settings"
    if [ "$DRY" -eq 1 ]; then
        echo "+ append to ~/.bashrc once:"
        settings_block | sed 's/^/    /'
    elif grep -q ">>> $MARK >>>" "$HOME/.bashrc" 2> /dev/null; then
        skipped "settings: block already in ~/.bashrc"
    else
        settings_block >> "$HOME/.bashrc"
        did "RECAL_* settings appended to ~/.bashrc"
    fi
    run mkdir -p "$DATA_DIR/third_party" "$DATA_DIR/mx" "$DATA_DIR/tmp"
    # a non-login shell, like this script, does not read ~/.bashrc
    export RECAL_STORE="s3://$BUCKET/$PREFIX" RECAL_REPO="$REPO_DIR" RECAL_THIRD_PARTY="$DATA_DIR/third_party"
    export RECAL_QUEUE_DIR="$DATA_DIR/mx" RECAL_TMP="$DATA_DIR/tmp"
}

missing_secret() {
    skipped "secret $1: not found; upload it with: scripts/upload_secrets.sh (target s3://$BUCKET/ops/secrets/$1)"
}

fetch_secret() {
    aws s3 cp "s3://$BUCKET/ops/secrets/$1" "$2" --only-show-errors > /dev/null 2>&1
}

step_secrets() {
    log "step secrets"
    if [ "$DRY" -eq 1 ]; then
        echo "+ aws s3 cp s3://$BUCKET/ops/secrets/env.local $REPO_DIR/.env.local; chmod 600"
        echo "+ aws s3 cp s3://$BUCKET/ops/secrets/github_token (git credential store, mode 600)"
        echo "+ aws s3 cp s3://$BUCKET/ops/secrets/git_identity (git user.name and user.email)"
        return 0
    fi
    local work
    work=$(mktemp -d)
    if fetch_secret env.local "$work/env.local"; then
        install -m 600 "$work/env.local" "$REPO_DIR/.env.local"
        did "secret env.local installed (mode 600)"
    else
        missing_secret env.local
    fi
    if fetch_secret github_token "$work/token"; then
        git config --global credential.helper store
        (umask 077; printf 'https://x-access-token:%s@github.com\n' "$(tr -d '\r\n' < "$work/token")" > "$HOME/.git-credentials")
        chmod 600 "$HOME/.git-credentials"
        did "secret github_token stored in ~/.git-credentials (mode 600)"
    else
        missing_secret github_token
    fi
    if fetch_secret git_identity "$work/identity"; then
        git config --global user.name "$(sed -n 1p "$work/identity")"
        git config --global user.email "$(sed -n 2p "$work/identity")"
        did "git identity set"
    else
        missing_secret git_identity
    fi
    rm -rf "$work"
}

step_context() {
    log "step agent context"
    if [ "$DRY" -eq 1 ]; then
        echo "+ aws s3 cp context/common.md and context/$DATASET.md (if present) -> ~/agent_context.md"
        return 0
    fi
    local work part found=0
    work=$(mktemp -d)
    : > "$work/context.md"
    for part in common "$DATASET"; do
        if aws s3 cp "s3://$BUCKET/ops/context/$part.md" "$work/$part.md" --only-show-errors > /dev/null 2>&1; then
            cat "$work/$part.md" >> "$work/context.md"
            printf '\n' >> "$work/context.md"
            found=1
        else
            skipped "context $part.md: not found at s3://$BUCKET/ops/context/$part.md"
        fi
    done
    if [ "$found" -eq 1 ]; then
        mv "$work/context.md" "$HOME/agent_context.md"
        did "agent context written to ~/agent_context.md"
    fi
    rm -rf "$work"
}

step_claude() {
    log "step Claude Code"
    if [ "$WITH_CLAUDE" -eq 0 ]; then skipped "claude code: --no-claude"; return 0; fi
    if [ "$DRY" -eq 0 ] && { command -v claude > /dev/null 2>&1 || [ -x "$HOME/.local/bin/claude" ]; }; then
        skipped "claude code: already installed"
    else
        if [ "$DRY" -eq 1 ]; then echo "+ curl -fsSL https://claude.ai/install.sh | bash"
        else curl -fsSL https://claude.ai/install.sh | bash; fi
        did "claude code installed"
    fi
    if [ "$DRY" -eq 1 ]; then
        echo "+ add ~/.local/bin to PATH in ~/.bashrc (once)"
    elif ! grep -q '\.local/bin' "$HOME/.bashrc" 2> /dev/null; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    fi
}

step_tmux() {
    log "step tmux"
    if [ "$WITH_TMUX" -eq 0 ]; then skipped "tmux session: --no-tmux"; return 0; fi
    local cmd="sudo -E bash scripts/ec2_bootstrap.sh --role prepare --datasets $DATASET 2>&1 | tee $DATA_DIR/bootstrap.log"
    if [ "$DRY" -eq 0 ] && tmux has-session -t main 2> /dev/null; then
        skipped "tmux: session main already exists"
        return 0
    fi
    run tmux new-session -d -s main -n bootstrap -c "$REPO_DIR" "$cmd; exec bash"
    run tmux new-window -t main -n agent -c "$REPO_DIR"
    did "tmux session main: window bootstrap running the prepare bootstrap, empty window agent"
}

final_message() {
    echo
    local mode=""
    [ "$DRY" -eq 0 ] || mode=" (dry run)"
    echo "== instance setup$mode for $DATASET =="
    echo "done:"
    printf '%s' "${DONE:-  (nothing)}"
    echo
    echo "skipped:"
    printf '%s' "${SKIPPED:-  (nothing)}"
    echo
    echo "left for you:"
    echo "  1. tmux attach -t main"
    echo "  2. in the agent window: claude --remote-control, then log in"
    echo "  3. paste ~/agent_context.md (or tell the agent to read it)"
}

main() {
    parse_args "$@"
    if [ "$DRY" -eq 0 ]; then
        [ "$(id -u)" -ne 0 ] || die "run as the normal user (ubuntu), not root; the script uses sudo where needed"
        command -v sudo > /dev/null 2>&1 || die "sudo is required"
    fi
    trap 'on_error $LINENO "$BASH_COMMAND"' ERR
    step_system
    step_volume
    step_code
    step_settings
    step_secrets
    step_context
    step_claude
    step_tmux
    final_message
}

if [ -z "${BASH_SOURCE[0]:-}" ] || [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
