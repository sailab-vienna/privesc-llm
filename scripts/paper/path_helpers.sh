#!/usr/bin/env bash

PAPER_HELPER_ROOT="${PAPER_HELPER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [ -z "${PAPER_PYTHON_USE_UV+x}" ]; then
  if command -v uv >/dev/null 2>&1 && [ -f "$PAPER_HELPER_ROOT/pyproject.toml" ]; then
    PAPER_PYTHON_USE_UV=1
  else
    PAPER_PYTHON_USE_UV=0
  fi
fi

paper_python() {
  if [ "$PAPER_PYTHON_USE_UV" = "1" ]; then
    (
      cd "$PAPER_HELPER_ROOT"
      if [ -n "${DATA:-}" ]; then
        export UV_CACHE_DIR="${UV_CACHE_DIR:-$DATA/privesc-llm/cache/uv}"
      fi
      PYTHONPATH="$PAPER_HELPER_ROOT${PYTHONPATH:+:$PYTHONPATH}" uv run --frozen --no-sync python "$@"
    )
  else
    PYTHONPATH="$PAPER_HELPER_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 "$@"
  fi
}

slugify() {
  paper_python - "$1" <<'PY'
import sys

from src.paths import path_slug

print(path_slug(sys.argv[1]), end="")
PY
}

model_slug() {
  paper_python - "$1" <<'PY'
import sys

from src.paths import base_model_slug

print(base_model_slug(sys.argv[1]), end="")
PY
}

paper_abs_path() {
  paper_python - "$1" <<'PY'
import sys
from pathlib import Path

print(Path(sys.argv[1]).expanduser().resolve(strict=False), end="")
PY
}

paper_assert_path_equals() {
  local actual="$1"
  local expected="$2"
  local message="$3"
  if [ "$(paper_abs_path "$actual")" != "$(paper_abs_path "$expected")" ]; then
    echo "[ERROR] $message" >&2
    echo "[ERROR] got:      $actual" >&2
    echo "[ERROR] expected: $expected" >&2
    return 1
  fi
}

paper_assert_path_under() {
  paper_python - "$1" "$2" "$3" <<'PY'
import sys
from pathlib import Path

actual = Path(sys.argv[1]).expanduser().resolve(strict=False)
parent = Path(sys.argv[2]).expanduser().resolve(strict=False)
message = sys.argv[3]

if actual != parent and parent not in actual.parents:
    raise SystemExit(
        f"[ERROR] {message}\n[ERROR] got:      {actual}\n[ERROR] expected under: {parent}"
    )
PY
}

paper_assert_slurm_export_value() {
  local name="$1"
  local value="$2"
  if [[ "$value" == *,* ]]; then
    echo "[ERROR] $name must not contain commas because Slurm --export is comma-separated: $value" >&2
    return 1
  fi
}

paper_eval_stage_slug() {
  local experiment="$1"
  if [[ "$experiment" == *paper_procedural* ]]; then
    printf '%s' "procedural"
  else
    printf '%s' "static"
  fi
}

paper_eval_artifact_path() {
  local eval_root="$1"
  local stage_slug="$2"
  local artifact_kind="$3"
  local artifact_slug="$4"
  printf '%s/%s/%s/%s' "${eval_root%/}" "$stage_slug" "$artifact_kind" "$artifact_slug"
}

paper_is_lora_adapter_dir() {
  local adapter_dir="$1"
  [ -d "$adapter_dir" ] \
    && [ -s "$adapter_dir/adapter_model.safetensors" ] \
    && [ -s "$adapter_dir/adapter_config.json" ]
}

paper_assert_lora_adapter_dir() {
  local adapter_dir="$1"
  local label="${2:-LoRA adapter}"
  if [ ! -d "$adapter_dir" ]; then
    echo "[ERROR] $label dir not found: $adapter_dir" >&2
    return 1
  fi
  if [ ! -s "$adapter_dir/adapter_model.safetensors" ]; then
    echo "[ERROR] $label missing non-empty adapter_model.safetensors: $adapter_dir" >&2
    return 1
  fi
  if [ ! -s "$adapter_dir/adapter_config.json" ]; then
    echo "[ERROR] $label missing non-empty adapter_config.json: $adapter_dir" >&2
    return 1
  fi
}

paper_run_root_path() {
  local prefix="$1"
  local experiment_id="$2"
  local base_model_slug="$3"
  local condition="$4"
  local run_id="$5"
  local rel
  rel="$(paper_python - "$experiment_id" "$base_model_slug" "$condition" "$run_id" <<'PY'
import sys

from src.paths import paper_run_root_from_slug

print(paper_run_root_from_slug(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]).as_posix(), end="")
PY
)"
  if [ -n "$prefix" ]; then
    printf '%s/%s' "${prefix%/}" "$rel"
  else
    printf '%s' "$rel"
  fi
}
