#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../artifact" && pwd)"
exec bash "$PROJECT_DIR/scripts/paper/run_artifact_eval.sh"
