#!/bin/bash
# Laptop-side: upload the instance secrets and agent context to the private bucket (see scripts/instance_setup.sh).
#
#   bash scripts/upload_secrets.sh --env-file local/private/drive_links.env --token-file ~/token.txt \
#       --name "Full Name" --email me@example.com [--context-dir local/private/context] [--bucket recal-eval-aa]
#
# Uploads, under s3://<bucket>/ops/: secrets/env.local, secrets/github_token, secrets/git_identity (name, then
# email) and context/*.md (common.md, <dataset>.md). Any subset of the options may be given. Refuses unless the
# bucket has all four Block Public Access settings on. Values are never printed. --dry-run lists the uploads.
set -eu -o pipefail

BUCKET="recal-eval-aa"
ENV_FILE=""
TOKEN_FILE=""
NAME=""
EMAIL=""
CONTEXT_DIR=""
DRY=0

die() { echo "[upload_secrets] $*" >&2; exit 1; }

parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --bucket) BUCKET="${2:?--bucket needs a name}"; shift ;;
            --env-file) ENV_FILE="${2:?--env-file needs a path}"; shift ;;
            --token-file) TOKEN_FILE="${2:?--token-file needs a path}"; shift ;;
            --name) NAME="${2:?--name needs a value}"; shift ;;
            --email) EMAIL="${2:?--email needs a value}"; shift ;;
            --context-dir) CONTEXT_DIR="${2:?--context-dir needs a path}"; shift ;;
            --dry-run) DRY=1 ;;
            *) die "unknown option: $1" ;;
        esac
        shift
    done
    if [ -z "$ENV_FILE$TOKEN_FILE$NAME$EMAIL$CONTEXT_DIR" ]; then die "nothing to upload (see the header for options)"; fi
    if [ -n "$NAME$EMAIL" ] && { [ -z "$NAME" ] || [ -z "$EMAIL" ]; }; then die "--name and --email go together"; fi
}

# The configuration of Block Public Access as JSON on stdin: true only when all four settings are true.
public_access_blocked() {
    local flags
    flags=$(tr -d ' \n' | grep -o '"\(BlockPublicAcls\|IgnorePublicAcls\|BlockPublicPolicy\|RestrictPublicBuckets\)":true' | wc -l)
    [ "$flags" -eq 4 ]
}

check_bucket() {
    local config
    config=$(aws s3api get-public-access-block --bucket "$BUCKET" --query PublicAccessBlockConfiguration --output json 2> /dev/null) \
        || die "cannot read the Block Public Access settings of $BUCKET; refusing to upload"
    if ! echo "$config" | public_access_blocked; then
        die "bucket $BUCKET does not have all four Block Public Access settings on; refusing to upload"
    fi
}

upload() {
    local source="$1" key="$2"
    [ -f "$source" ] || die "$source does not exist"
    if [ "$DRY" -eq 1 ]; then echo "would upload a file to s3://$BUCKET/$key"; return 0; fi
    aws s3 cp "$source" "s3://$BUCKET/$key" --sse AES256 --only-show-errors > /dev/null
    echo "uploaded s3://$BUCKET/$key"
}

main() {
    parse_args "$@"
    [ "$DRY" -eq 1 ] || check_bucket
    [ -z "$ENV_FILE" ] || upload "$ENV_FILE" ops/secrets/env.local
    [ -z "$TOKEN_FILE" ] || upload "$TOKEN_FILE" ops/secrets/github_token
    if [ -n "$NAME" ]; then
        local tmp
        tmp=$(mktemp)
        (umask 077; printf '%s\n%s\n' "$NAME" "$EMAIL" > "$tmp")
        upload "$tmp" ops/secrets/git_identity
        rm -f "$tmp"
    fi
    if [ -n "$CONTEXT_DIR" ]; then
        local f
        for f in "$CONTEXT_DIR"/*.md; do
            [ -e "$f" ] || die "no .md files in $CONTEXT_DIR"
            upload "$f" "ops/context/$(basename "$f")"
        done
    fi
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
