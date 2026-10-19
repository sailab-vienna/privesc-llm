#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../artifact" && pwd)"
cd "$PROJECT_DIR"
if [ ! -x .venv/bin/python ]; then
  printf '%s\n' 'FAIL: run bash install.sh from the repository root first.' >&2
  exit 1
fi
exec .venv/bin/python scripts/paper/verify_artifact.py "$@"
