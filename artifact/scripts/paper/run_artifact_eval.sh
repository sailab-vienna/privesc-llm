#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

RUNS_PER_SCENARIO="${RUNS_PER_SCENARIO:-3}"
SCENARIO_BACKEND="${PRIVESC_SCENARIO_BACKEND:-local_docker}"
SCENARIO_COUNT=12
MODEL_COUNT=3
WORKERS="${EVAL_RUNNER_WORKERS:-8}"
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
BASE_REVISION="cdbee75f17c01a7cc42f958dc650907174af0554"
MODEL_ARTIFACT="sailab-vienna/privesc-llm-4b"
MODEL_ARTIFACT_REVISION="e7a10aadbcc0d0cd721533480d4fd47ee083dc63"
TOTAL_STAGES=8

if [ -t 1 ] && [ "${TERM:-dumb}" != "dumb" ] && [ -z "${NO_COLOR+x}" ]; then
  CYAN=$'\033[1;36m'
  BOLD=$'\033[1m'
  GREEN=$'\033[1;32m'
  RED=$'\033[1;31m'
  RESET=$'\033[0m'
else
  CYAN="" BOLD="" GREEN="" RED="" RESET=""
fi

stage() {
  printf '\n%b\n' "${CYAN}[$1/$TOTAL_STAGES]${RESET} $2"
}

info() {
  printf '  %s\n' "$*"
}

fail() {
  printf '%b\n' "${RED}FAIL:${RESET} $*" >&2
  exit 1
}

case "$SCENARIO_BACKEND" in
  local_docker) ;;
  remote_ssh)
    [[ "${PRIVESC_SSH_SERVERS:-}" =~ ^[^,:[:space:]]+:[0-9]{1,5}$ ]] ||
      fail "remote_ssh requires one PRIVESC_SSH_SERVERS host:port endpoint"
    [[ -f "${PRIVESC_KEY:-}" && -r "${PRIVESC_KEY:-}" ]] ||
      fail "remote_ssh requires PRIVESC_KEY to name a readable private key"
    export PRIVESC_USER="${PRIVESC_USER:-root}"
    export DOCKER_HOST="ssh://${PRIVESC_USER}@${PRIVESC_SSH_SERVERS}"
    export PAPER_EVAL_SKIP_DOTENV=1
    ;;
  *) fail "PRIVESC_SCENARIO_BACKEND must be local_docker or remote_ssh" ;;
esac

if [[ ! "$RUNS_PER_SCENARIO" =~ ^[1-9][0-9]*$ ]]; then
  fail "RUNS_PER_SCENARIO must be a positive integer without leading zeros; got: $RUNS_PER_SCENARIO"
fi
EXPECTED_RUNS=$((SCENARIO_COUNT * RUNS_PER_SCENARIO))
TOTAL_RUNS=$((MODEL_COUNT * EXPECTED_RUNS))

run_logged() {
  local label="$1"
  local log="$2"
  shift 2
  if "$@" >"$log" 2>&1; then
    return
  fi
  printf '%b\n' "${RED}FAIL:${RESET} $label failed. Last 40 log lines:" >&2
  tail -n 40 "$log" >&2
  printf 'Full log: %s\n' "$log" >&2
  exit 1
}

for required in git uv docker jq curl nvidia-smi; do
  command -v "$required" >/dev/null 2>&1 ||
    fail "missing required command: $required"
done
docker info >/dev/null 2>&1 ||
  fail "Docker is unavailable. Check the daemon and connection, then rerun."
nvidia-smi >/dev/null 2>&1 ||
  fail "The NVIDIA driver is unavailable. Check the GPU driver and rerun."

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="${PAPER_AE_OUTPUT_ROOT:-$PROJECT_DIR/outputs/ae/live/$RUN_ID}"
MODEL_DIR="${PAPER_AE_MODEL_DIR:-$PROJECT_DIR/outputs/ae/privesc-llm-4b}"
if [ -e "$OUTPUT_ROOT" ]; then
  fail "output directory already exists: $OUTPUT_ROOT"
fi
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR" "$MODEL_DIR"

SOURCE_REVISION="$(git rev-parse HEAD)"
printf '\n%b\n' "${BOLD}PrivEsc-LLM ACSAC live artifact evaluation${RESET}"
info "Source revision: $SOURCE_REVISION"
info "Base model: $BASE_MODEL@$BASE_REVISION"
info "Adapters: $MODEL_ARTIFACT@$MODEL_ARTIFACT_REVISION"
info "Protocol: $MODEL_COUNT conditions x $SCENARIO_COUNT scenarios x $RUNS_PER_SCENARIO runs = $TOTAL_RUNS; $WORKERS workers; r20 primary; r60 maximum"

stage 1 "Initialize pinned source dependencies"
run_logged "submodule initialization" "$LOG_DIR/01-submodules.log" \
  git submodule update --init --recursive

stage 2 "Install the locked analysis environment"
run_logged "dependency installation" "$LOG_DIR/02-install.log" \
  bash scripts/paper/install_analysis.sh

stage 3 "Build benchmark images and verify the $SCENARIO_COUNT paper scenarios"
run_logged "benchmark patch" "$LOG_DIR/03-patch.log" \
  bash scripts/apply_benchmark_patch.sh
run_logged "benchmark image build" "$LOG_DIR/03-build.log" \
  bash external/benchmark-privesc-linux/docker/build.sh
run_logged "benchmark container verification" "$LOG_DIR/03-test.log" \
  env PRIVESC_SCENARIO_BACKEND="$SCENARIO_BACKEND" \
  uv run --frozen python -m pytest -q test/test_solutions.py

stage 4 "Download the pinned SFT and RL adapters"
run_logged "adapter download" "$LOG_DIR/04-download.log" \
  env HF_HUB_DISABLE_PROGRESS_BARS=1 \
  uv run --frozen hf download "$MODEL_ARTIFACT" \
  --revision "$MODEL_ARTIFACT_REVISION" \
  --local-dir "$MODEL_DIR"

export PAPER_EVAL_CACHE_ROOT="${PAPER_EVAL_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/privesc-llm-ae}"
export PAPER_EVAL_UV_PROJECT_ENVIRONMENT="${PAPER_EVAL_UV_PROJECT_ENVIRONMENT:-$PROJECT_DIR/outputs/ae/vllm-env}"
export PAPER_EVAL_UV_GROUP=rl
export PAPER_EVAL_EXPERIMENT=eval/paper_static_qwen3_4b_base
export PAPER_EVAL_MODEL_REVISION="$BASE_REVISION"
export PAPER_EVAL_HYDRA_OVERRIDES="scenario.backend=$SCENARIO_BACKEND"
export PAPER_EVAL_VLLM_MAX_MODEL_LEN=32768
export PAPER_EVAL_VLLM_TOOL_CALL_PARSER=hermes
export PAPER_EVAL_VLLM_HEALTH_CHECK_ATTEMPTS=360
export EVAL_RUNNER_RUNS_PER_ITEM="$RUNS_PER_SCENARIO"
export EVAL_ANALYZE_EXPECTED_RUNS_PER_SCENARIO="$RUNS_PER_SCENARIO"
export EVAL_RUNNER_WORKERS="$WORKERS"
export EVAL_RUNNER_RUN_WALL_CLOCK_TIMEOUT="${EVAL_RUNNER_RUN_WALL_CLOCK_TIMEOUT:-7200}"

stage 5 "Evaluate Base: $EXPECTED_RUNS runs (log: $LOG_DIR/05-base.log)"
run_logged "Base evaluation" "$LOG_DIR/05-base.log" \
  env PAPER_EVAL_RUN_ROOT="$OUTPUT_ROOT/base" \
  PAPER_EVAL_ENABLE_LORA=0 \
  PAPER_EVAL_LORA_ADAPTER_DIR= \
  bash scripts/run_model_eval_vllm.sh \
  "$BASE_MODEL" qwen3_4b_base

stage 6 "Evaluate SFT: $EXPECTED_RUNS runs (log: $LOG_DIR/06-sft.log)"
run_logged "SFT evaluation" "$LOG_DIR/06-sft.log" \
  env PAPER_EVAL_RUN_ROOT="$OUTPUT_ROOT/sft" \
  PAPER_EVAL_ENABLE_LORA=1 \
  PAPER_EVAL_LORA_ADAPTER_DIR="$MODEL_DIR/sft_adapter" \
  PAPER_EVAL_LORA_MODEL_NAME=qwen3_4b_sft \
  bash scripts/run_model_eval_vllm.sh \
  "$BASE_MODEL" qwen3_4b_sft

stage 7 "Evaluate PrivEsc-LLM: $EXPECTED_RUNS runs (log: $LOG_DIR/07-privesc.log)"
run_logged "PrivEsc-LLM evaluation" "$LOG_DIR/07-privesc.log" \
  env PAPER_EVAL_RUN_ROOT="$OUTPUT_ROOT/rl" \
  PAPER_EVAL_ENABLE_LORA=1 \
  PAPER_EVAL_LORA_ADAPTER_DIR="$MODEL_DIR/rl_adapter" \
  PAPER_EVAL_LORA_MODEL_NAME=privesc_llm_4b \
  bash scripts/run_model_eval_vllm.sh \
  "$BASE_MODEL" privesc_llm_4b

SUMMARY_FILES=(
  "$OUTPUT_ROOT/base/eval/static/raw/final/stats/evaluation_summary_paper.json"
  "$OUTPUT_ROOT/sft/eval/static/raw/final/stats/evaluation_summary_paper.json"
  "$OUTPUT_ROOT/rl/eval/static/raw/final/stats/evaluation_summary_paper.json"
)

stage 8 "Verify run counts and the Base-to-SFT-to-PrivEsc improvement"
for summary in "${SUMMARY_FILES[@]}"; do
  [ -f "$summary" ] || fail "missing evaluation summary: $summary"
done

if ! jq -s -e \
  --argjson models "$MODEL_COUNT" \
  --argjson scenarios "$SCENARIO_COUNT" \
  --argjson total "$EXPECTED_RUNS" \
  --argjson per "$RUNS_PER_SCENARIO" '
    length == $models
    and all(.[];
      [.scenario_model_stats[][] | .num_runs] as $scenario_runs
      | (.model_stats | length) == 1
      and ([.model_stats[].num_runs] == [$total])
      and (.scenario_model_stats | length) == $scenarios
      and ($scenario_runs | length) == $scenarios
      and all($scenario_runs[]; . == $per)
    )
  ' "${SUMMARY_FILES[@]}" >/dev/null; then
  fail "run-count check failed. Expected $SCENARIO_COUNT scenarios x $RUNS_PER_SCENARIO eligible runs and $EXPECTED_RUNS total runs per model."
fi

if ! RESULTS="$(
  jq -s -r '
    def percent:
      . * 1000 | round
      | "\(. / 10 | floor).\(. % 10)%";

    {
      "qwen3_4b_base": "Qwen3 4B",
      "qwen3_4b_sft": "Qwen3 4B SFT",
      "privesc_llm_4b": "PrivEsc-LLM 4B"
    } as $names
    | .[].model_stats
    | to_entries[]
    | ($names[.key] // .key) as $model
    | .value
    | .num_runs as $runs
    | [.sr_within_rounds_10, .sr_within_rounds_20, .sr_within_rounds_60] as $rates
    | ($rates | map(. * $runs | round | tostring) | join("/")) as $successes
    | ($rates | map(percent) | join("/")) as $percentages
    | "  \($model): N=\($runs); r10/r20/r60=\($successes) (\($percentages))"
  ' "${SUMMARY_FILES[@]}" 2>&1
)"; then
  fail "result rendering failed. Expected numeric num_runs and r10/r20/r60 rates in every summary. $RESULTS"
fi
printf '%b\n' "${BOLD}Results${RESET}"
printf '%s\n' "$RESULTS"

if ! jq -s -e '
  [.[].model_stats[].success_rate_within_primary_budget] as [$base, $sft, $rl]
  | $sft >= 0.60
    and $rl >= 0.80
    and $sft >= 1.5 * $base
    and $rl - $sft >= 0.05
' "${SUMMARY_FILES[@]}" >/dev/null; then
  fail "effect check failed. Expected SFT >=60% and >=1.5x Base, plus PrivEsc-LLM >=80% and >=5 percentage points above SFT at r20."
fi

info "Outputs: $OUTPUT_ROOT"
info "Logs: $LOG_DIR"
printf '\n%b\n' "${GREEN}PASS:${RESET} $TOTAL_RUNS-run live evaluation met the Base-to-SFT-to-PrivEsc r20 gates"
