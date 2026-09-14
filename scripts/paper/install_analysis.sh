#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

if [ -t 1 ] && [ "${TERM:-dumb}" != "dumb" ] && [ -z "${NO_COLOR+x}" ]; then
  CYAN=$'\033[1;36m'
  RED=$'\033[1;31m'
  RESET=$'\033[0m'
else
  CYAN="" RED="" RESET=""
fi

printf '%b\n' "${CYAN}[1/1]${RESET} Install the locked analysis environment"
command -v uv >/dev/null 2>&1 || {
  printf '%b\n' "${RED}FAIL:${RESET} uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
}
if ! uv sync --frozen --group dev --no-install-project --quiet; then
  printf '%b\n' "${RED}FAIL:${RESET} dependency installation failed" >&2
  exit 1
fi

printf '%s\n' "Installed locked analysis environment"
