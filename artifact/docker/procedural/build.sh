#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-privesc_procedural}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

docker build -t "${IMAGE}:latest" "${SCRIPT_DIR}"
echo "[OK] Built ${IMAGE}:latest"
