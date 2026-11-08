#!/usr/bin/env bash
# Check out an exact commit on the instance, run preflight, build the drivers.
# No development happens on the instance: it only ever runs a clean checkout.
#   usage: deploy.sh <repo-url> <commit> <checkout-dir> [data-dir]
set -euo pipefail

# A private repo or an unknown host key must fail, not prompt.
export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -o BatchMode=yes -o ConnectTimeout=30}"

repo_url="${1:?repo url}"
commit="${2:?commit sha}"
dest="${3:?checkout dir}"
data_dir="${4:-/data}"

if [ -e "$dest" ] && [ -n "$(git -C "$dest" status --porcelain 2>/dev/null)" ]; then
    echo "deploy.sh: $dest has local changes; refusing to run from a dirty tree" >&2
    exit 1
fi
[ -d "$dest/.git" ] || git clone "$repo_url" "$dest"
git -C "$dest" fetch origin
git -C "$dest" checkout --detach "$commit"
if [ -n "$(git -C "$dest" status --porcelain)" ]; then
    echo "deploy.sh: dirty tree after checkout" >&2
    exit 1
fi

cd "$dest"
PYTHONPATH=. python3 -m harness.deploy.preflight --data-dir "$data_dir"
bash harness/deploy/build_linux.sh
echo "deployed $(git rev-parse HEAD)"
