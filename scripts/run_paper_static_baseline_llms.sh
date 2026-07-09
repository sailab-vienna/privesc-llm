#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <qwen_model_ref> [project_dir]" >&2
  echo "Example: $0 Qwen/Qwen3-4B-Instruct-2507" >&2
}

if [ $# -lt 1 ]; then
  usage
  exit 1
fi

QWEN_MODEL_REF="$1"
PROJECT_DIR="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUTPUT_ROOT="${PAPER_BASELINE_OUTPUT_ROOT:-outputs/evals/paper_baselines/static_baseline_llms}"
SCENARIO_BACKEND="${PAPER_EVAL_SCENARIO_BACKEND:-local_docker}"

if [ ! -d "$PROJECT_DIR" ]; then
  echo "[ERROR] project directory not found: $PROJECT_DIR" >&2
  exit 1
fi

require_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[ERROR] required command not found in PATH: $cmd" >&2
    exit 1
  fi
}

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "[ERROR] required environment variable is not set: $name" >&2
    exit 1
  fi
}

check_benchmark_images() {
  local missing=()
  local scenario
  for scenario in \
    01_vuln_suid_gtfo \
    02_vuln_password_in_shell_history \
    03_vuln_sudo_no_password \
    05_vuln_sudo_gtfo \
    06_vuln_docker \
    07_root_password_reuse_mysql \
    08_root_password_reuse \
    09_root_password_root \
    10_root_allows_lowpriv_to_ssh \
    11_cron_calling_user_wildcard \
    12_cron_calling_user_file \
    13_file_with_root_password
  do
    if ! docker image inspect "privesc_${scenario}:latest" >/dev/null 2>&1; then
      missing+=("privesc_${scenario}:latest")
    fi
  done

  if [ ${#missing[@]} -gt 0 ]; then
    echo "[ERROR] missing benchmark Docker images:" >&2
    printf '  - %s\n' "${missing[@]}" >&2
    echo "[ERROR] build them with: bash external/benchmark-privesc-linux/docker/build.sh" >&2
    exit 1
  fi
}

run_api_experiment() {
  local experiment="$1"
  local model_dir="$2"

  echo "[INFO] Running $experiment -> $OUTPUT_ROOT/traces/$model_dir"
  uv run python -m src.runner \
    "+experiment=$experiment" \
    runner.output_dir="$OUTPUT_ROOT"

  if [ ! -d "$OUTPUT_ROOT/traces/$model_dir" ]; then
    echo "[ERROR] expected trace directory not found: $OUTPUT_ROOT/traces/$model_dir" >&2
    exit 1
  fi
}

cd "$PROJECT_DIR"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

require_command uv
require_env OPENAI_API_BASE
require_env OPENAI_API_KEY
if [ "$SCENARIO_BACKEND" = "local_docker" ]; then
  require_command docker
  check_benchmark_images
fi
export PRIVESC_SCENARIO_BACKEND="$SCENARIO_BACKEND"

echo "[INFO] Using scenario backend: $SCENARIO_BACKEND"

mkdir -p "$OUTPUT_ROOT"
if [ -d "$OUTPUT_ROOT/traces" ]; then
  echo "[WARN] existing traces found under $OUTPUT_ROOT; runner will resume completed benchmark-eligible runs"
fi

run_api_experiment "eval/paper_static_opus47" "claude-opus-4.7"
run_api_experiment "eval/paper_static_deepseek32" "deepseek-v3.2"

echo "[INFO] Running local Qwen3-4B baseline via vLLM -> $OUTPUT_ROOT/traces/qwen3-4b"
PAPER_EVAL_EXPERIMENT=eval/paper_static_qwen3_4b_base \
PAPER_EVAL_ALLOW_EXTERNAL_OUTPUT=1 \
PAPER_EVAL_OUTPUT_ROOT="$OUTPUT_ROOT" \
PAPER_EVAL_OUTPUT_DIR="$OUTPUT_ROOT" \
EVAL_ANALYZE_ALL_MODELS=0 \
  bash scripts/run_model_eval_vllm.sh \
  "$QWEN_MODEL_REF" \
  "qwen3-4b" \
  "$PROJECT_DIR"

for trace_dir in \
  "$OUTPUT_ROOT/traces/claude-opus-4.7" \
  "$OUTPUT_ROOT/traces/deepseek-v3.2" \
  "$OUTPUT_ROOT/traces/qwen3-4b"
do
  if [ ! -d "$trace_dir" ]; then
    echo "[ERROR] missing trace directory: $trace_dir" >&2
    exit 1
  fi
done

if [ ! -f "$OUTPUT_ROOT/stats/evaluation_summary.json" ]; then
  echo "[ERROR] missing analysis summary: $OUTPUT_ROOT/stats/evaluation_summary.json" >&2
  exit 1
fi

if [ ! -f "$OUTPUT_ROOT/stats/evaluation_summary_paper.json" ]; then
  echo "[ERROR] missing paper analysis summary: $OUTPUT_ROOT/stats/evaluation_summary_paper.json" >&2
  exit 1
fi

echo "[INFO] Paper static baseline LLM reevaluation finished"
echo "[INFO] Output root: $OUTPUT_ROOT"
