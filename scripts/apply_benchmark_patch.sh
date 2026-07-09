#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
SUBMODULE_DIR="$REPO_ROOT/external/benchmark-privesc-linux"
PATCH_PATH="$REPO_ROOT/patches/benchmark-privesc-linux-source-delta.patch"

if [ ! -d "$SUBMODULE_DIR/.git" ] && [ ! -f "$SUBMODULE_DIR/.git" ]; then
  echo "[ERROR] Missing submodule checkout: $SUBMODULE_DIR" >&2
  exit 1
fi

if git -C "$SUBMODULE_DIR" apply --reverse --check "$PATCH_PATH" >/dev/null 2>&1; then
  echo "[OK] Benchmark patch already applied"
  exit 0
fi

git -C "$SUBMODULE_DIR" apply --check "$PATCH_PATH"
git -C "$SUBMODULE_DIR" apply "$PATCH_PATH"
echo "[OK] Applied benchmark patch"
