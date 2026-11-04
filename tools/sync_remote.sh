#!/usr/bin/env bash
# Copy this checkout to the H100 box and install it editable there.
#   tools/sync_remote.sh            # sync + pip install -e
#   SKIP_INSTALL=1 tools/sync_remote.sh
set -euo pipefail

HOST=${GREEN_REMOTE_HOST:-root@103.73.232.171}
PORT=${GREEN_REMOTE_PORT:-64061}
KEY=${GREEN_REMOTE_KEY:-$HOME/.ssh/id_rsa}
DST=${GREEN_REMOTE_DIR:-/network-volume/rnd-sglang/green/GreenOptimizer}
REPO=$(cd "$(dirname "$0")/.." && pwd)

ssh_remote() { ssh -o BatchMode=yes -p "$PORT" -i "$KEY" "$HOST" "$@"; }

COPYFILE_DISABLE=1 tar -C "$REPO" --no-xattrs -cz --exclude .git --exclude '._*' \
    --exclude '__pycache__' --exclude '*.so' . |
    ssh_remote "mkdir -p '$DST' && tar -xz -C '$DST'"

if [ -z "${SKIP_INSTALL:-}" ]; then
  ssh_remote "cd '$DST' && /usr/bin/python -m pip install -q -e . --no-deps --no-build-isolation"
fi
echo "synced $REPO -> $HOST:$DST"
