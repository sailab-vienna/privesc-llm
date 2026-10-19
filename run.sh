#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$PROJECT_DIR/install.sh"
bash "$PROJECT_DIR/claims/claim1/run.sh"
bash "$PROJECT_DIR/claims/claim2/run.sh"
printf '\nPASS: archived evidence and live reproduction checks passed\n'
