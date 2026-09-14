#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <model_ref> <model_tag> [project_dir]" >&2
  echo "Example: $0 Qwen/Qwen3-4B-Instruct-2507 qwen3-4b" >&2
  echo "Note: model_tag becomes the eval model id unless PAPER_EVAL_MODEL_ID is set." >&2
  exit 1
fi

MODEL_REF="$1"
MODEL_TAG="$2"
PROJECT_DIR="${3:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [ -d "$MODEL_REF" ] && [ ! -f "$MODEL_REF/config.json" ]; then
  echo "[ERROR] local model path missing config.json: $MODEL_REF" >&2
  exit 1
fi

if [ ! -d "$PROJECT_DIR" ]; then
  echo "[ERROR] project directory not found: $PROJECT_DIR" >&2
  exit 1
fi

if [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
  export PATH="$PROJECT_DIR/.venv/bin:$PATH"
fi

. "$PROJECT_DIR/scripts/paper/path_helpers.sh"

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv not found in PATH" >&2
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "[ERROR] jq not found in PATH" >&2
  exit 1
fi

cd "$PROJECT_DIR"

if [ -f .env ] && [ "${PAPER_EVAL_SKIP_DOTENV:-0}" != 1 ]; then
  set -a
  source .env
  set +a
fi

CACHE_ROOT="${PAPER_EVAL_CACHE_ROOT:-}"
if [ -z "$CACHE_ROOT" ]; then
  if [ -n "${DATA:-}" ]; then
    CACHE_ROOT="$DATA/privesc-llm/cache"
  else
    CACHE_ROOT="$HOME/.cache/privesc-llm"
  fi
fi
HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CACHE_ROOT/vllm}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$CACHE_ROOT}"
FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
export HF_HOME HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE XDG_CACHE_HOME
export VLLM_CACHE_ROOT TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR FLASHINFER_WORKSPACE_BASE FLASHINFER_CACHE_DIR UV_CACHE_DIR
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$FLASHINFER_CACHE_DIR" "$UV_CACHE_DIR"
UV_GROUP="${PAPER_EVAL_UV_GROUP:-}"
UV_WITH="${PAPER_EVAL_UV_WITH:-}"
PATCH_VLLM_GEMMA4_BNB="${PAPER_EVAL_PATCH_VLLM_GEMMA4_BNB:-0}"
if [ -n "${PAPER_EVAL_UV_PROJECT_ENVIRONMENT:-}" ]; then
  export UV_PROJECT_ENVIRONMENT="$PAPER_EVAL_UV_PROJECT_ENVIRONMENT"
elif [ -n "$UV_GROUP" ]; then
  if [ -z "${DATA:-}" ]; then
    echo "[ERROR] PAPER_EVAL_UV_GROUP requires PAPER_EVAL_UV_PROJECT_ENVIRONMENT when DATA is unset" >&2
    exit 1
  fi
  export UV_PROJECT_ENVIRONMENT="$DATA/privesc-llm/envs/$UV_GROUP"
fi
if [ -n "$UV_GROUP" ] || [ "$PATCH_VLLM_GEMMA4_BNB" = "1" ]; then
  if [ -z "${UV_PROJECT_ENVIRONMENT:-}" ]; then
    echo "[ERROR] Gemma4/vLLM eval isolation requires UV_PROJECT_ENVIRONMENT" >&2
    exit 1
  fi
  PROJECT_VENV_REALPATH="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$PROJECT_DIR/.venv")"
  UV_ENV_REALPATH="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$UV_PROJECT_ENVIRONMENT")"
  if [ "$UV_ENV_REALPATH" = "$PROJECT_VENV_REALPATH" ]; then
    echo "[ERROR] Refusing to sync or patch the project .venv; set PAPER_EVAL_UV_PROJECT_ENVIRONMENT to an isolated path" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$UV_PROJECT_ENVIRONMENT")"
fi
echo "[INFO] cache_root=$CACHE_ROOT"
echo "[INFO] hf_home=$HF_HOME"
echo "[INFO] vllm_cache_root=$VLLM_CACHE_ROOT"
echo "[INFO] torchinductor_cache=$TORCHINDUCTOR_CACHE_DIR"
echo "[INFO] flashinfer_cache=$FLASHINFER_CACHE_DIR"
if [ -n "$UV_GROUP" ]; then
  echo "[INFO] uv_group=$UV_GROUP"
  echo "[INFO] uv_project_environment=$UV_PROJECT_ENVIRONMENT"
  uv sync --frozen --no-default-groups --group "$UV_GROUP"
fi
if [ -n "$UV_WITH" ]; then
  echo "[INFO] uv_with=$UV_WITH"
fi

UV_RUN=(uv run --frozen --no-sync)
if [ -n "$UV_GROUP" ]; then
  UV_RUN+=(--group "$UV_GROUP")
fi
if [ -n "$UV_WITH" ]; then
  UV_RUN+=(--with "$UV_WITH")
fi
if [ "$PATCH_VLLM_GEMMA4_BNB" = "1" ]; then
  "${UV_RUN[@]}" python scripts/patch_vllm_gemma4_bnb.py
fi

EVAL_EXPERIMENT="${PAPER_EVAL_EXPERIMENT:-eval/paper_static_qwen3_4b_base}"
BASE_MODEL_SLUG="${BASE_MODEL_SLUG:-$(model_slug "$MODEL_REF")}"
CONDITION_SLUG="${PAPER_EVAL_CONDITION_SLUG:-$(slugify "$MODEL_TAG")}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
PAPER_RUN_EXPERIMENT_ID="${PAPER_EVAL_RUN_EXPERIMENT_ID:-04_static_benchmark}"
DEFAULT_EVAL_STAGE_SLUG="$(paper_eval_stage_slug "$EVAL_EXPERIMENT")"
EVAL_STAGE_SLUG="${PAPER_EVAL_STAGE_SLUG:-$DEFAULT_EVAL_STAGE_SLUG}"
CANONICAL_EVAL_RUN_ROOT="$(paper_run_root_path "" "$PAPER_RUN_EXPERIMENT_ID" "$BASE_MODEL_SLUG" "$CONDITION_SLUG" "$RUN_ID")"
EVAL_RUN_ROOT="${PAPER_EVAL_RUN_ROOT:-$CANONICAL_EVAL_RUN_ROOT}"
EVAL_OUTPUT_ROOT="${PAPER_EVAL_OUTPUT_ROOT:-$EVAL_RUN_ROOT/eval}"
if [ -n "${PAPER_EVAL_OUTPUT_DIR:-}" ]; then
  EVAL_OUTPUT_DIR="$PAPER_EVAL_OUTPUT_DIR"
else
  EVAL_OUTPUT_DIR="$(paper_eval_artifact_path "$EVAL_OUTPUT_ROOT" "$EVAL_STAGE_SLUG" raw final)"
fi
if [ "${PAPER_EVAL_ALLOW_EXTERNAL_OUTPUT:-0}" != "1" ] && { [ -n "${PAPER_EVAL_OUTPUT_ROOT:-}" ] || [ -n "${PAPER_EVAL_OUTPUT_DIR:-}" ]; }; then
  paper_assert_path_under \
    "$EVAL_OUTPUT_DIR" \
    "$EVAL_RUN_ROOT/eval" \
    "PAPER_EVAL_OUTPUT_ROOT/PAPER_EVAL_OUTPUT_DIR must stay under the run-local eval directory."
fi
EVAL_RUNNER_WORKERS="${EVAL_RUNNER_WORKERS:-}"
EVAL_RUNNER_RUNS_PER_ITEM="${EVAL_RUNNER_RUNS_PER_ITEM:-}"
EVAL_RUNNER_MAX_RETRIES_PER_RUN="${EVAL_RUNNER_MAX_RETRIES_PER_RUN:-}"
EVAL_RUNNER_MAX_RUNS="${EVAL_RUNNER_MAX_RUNS:-}"
EVAL_RUNNER_RUN_WALL_CLOCK_TIMEOUT="${EVAL_RUNNER_RUN_WALL_CLOCK_TIMEOUT:-}"
EVAL_AGENT_MAX_TOKENS="${EVAL_AGENT_MAX_TOKENS:-}"
EVAL_SCENARIO_MAX_PARALLEL_TOOL_CALLS="${EVAL_SCENARIO_MAX_PARALLEL_TOOL_CALLS:-}"
PAPER_EVAL_HYDRA_OVERRIDES="${PAPER_EVAL_HYDRA_OVERRIDES:-}"
EVAL_ANALYZE_ALLOW_INCOMPLETE_RUNS="${EVAL_ANALYZE_ALLOW_INCOMPLETE_RUNS:-0}"
EVAL_ANALYZE_EXPECTED_RUNS_PER_SCENARIO="${EVAL_ANALYZE_EXPECTED_RUNS_PER_SCENARIO:-}"
EVAL_MODEL_ID_OVERRIDE="${PAPER_EVAL_MODEL_ID:-}"
VLLM_MODEL_REVISION="${PAPER_EVAL_MODEL_REVISION:-}"
VLLM_MAX_MODEL_LEN="${PAPER_EVAL_VLLM_MAX_MODEL_LEN:-32768}"
VLLM_GPU_MEMORY_UTILIZATION="${PAPER_EVAL_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_NUM_SEQS="${PAPER_EVAL_VLLM_MAX_NUM_SEQS:-64}"
VLLM_MAX_NUM_BATCHED_TOKENS="${PAPER_EVAL_VLLM_MAX_NUM_BATCHED_TOKENS:-}"
if [ -z "${PAPER_EVAL_VLLM_TOOL_CALL_PARSER:-}" ]; then
  case "$MODEL_REF" in
    *gemma4*|*gemma-4*) VLLM_TOOL_CALL_PARSER=gemma4 ;;
    *) VLLM_TOOL_CALL_PARSER=hermes ;;
  esac
else
  VLLM_TOOL_CALL_PARSER="$PAPER_EVAL_VLLM_TOOL_CALL_PARSER"
fi
if [ "$VLLM_TOOL_CALL_PARSER" = "none" ]; then
  VLLM_TOOL_CALL_PARSER=""
fi
VLLM_REASONING_PARSER="${PAPER_EVAL_VLLM_REASONING_PARSER-}"
VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS="${PAPER_EVAL_VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS:-}"
VLLM_MAX_LORA_RANK="${PAPER_EVAL_VLLM_MAX_LORA_RANK:-64}"
VLLM_ENFORCE_EAGER="${PAPER_EVAL_VLLM_ENFORCE_EAGER:-0}"
VLLM_TP="${PAPER_EVAL_VLLM_TP:-}"
VLLM_DP="${PAPER_EVAL_VLLM_DP:-}"
DEPLOYMENT_GPUS_PER_NODE="${PAPER_EVAL_DEPLOYMENT_GPUS_PER_NODE:-}"
LORA_ADAPTER_DIR="${PAPER_EVAL_LORA_ADAPTER_DIR:-}"
LORA_MODEL_NAME="${PAPER_EVAL_LORA_MODEL_NAME:-${MODEL_TAG}_lora}"
if [ -n "$LORA_ADAPTER_DIR" ]; then
  ENABLE_LORA="${PAPER_EVAL_ENABLE_LORA:-1}"
else
  ENABLE_LORA="${PAPER_EVAL_ENABLE_LORA:-0}"
fi

if [ "$ENABLE_LORA" = "1" ]; then
  if [ -z "$LORA_ADAPTER_DIR" ]; then
    echo "[ERROR] PAPER_EVAL_ENABLE_LORA=1 requires PAPER_EVAL_LORA_ADAPTER_DIR" >&2
    exit 1
  fi
  paper_assert_lora_adapter_dir "$LORA_ADAPTER_DIR" "LoRA adapter"
  EVAL_MODEL_ID="$LORA_MODEL_NAME"
else
  if [ -d "$MODEL_REF" ]; then
    EVAL_MODEL_ID="$MODEL_REF"
  else
    EVAL_MODEL_ID="$MODEL_TAG"
  fi
fi
if [ -n "$EVAL_MODEL_ID_OVERRIDE" ]; then
  EVAL_MODEL_ID="$EVAL_MODEL_ID_OVERRIDE"
fi

get_free_port() {
  python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()'
}

VLLM_PORT="$(get_free_port)"
VLLM_ENDPOINT="http://127.0.0.1:${VLLM_PORT}"
VLLM_API_BASE="${VLLM_ENDPOINT}/v1"
VLLM_HEALTH_URL="${VLLM_ENDPOINT}/health"
VLLM_HEALTH_CHECK_ATTEMPTS="${PAPER_EVAL_VLLM_HEALTH_CHECK_ATTEMPTS:-60}"
VLLM_HEALTH_CHECK_INTERVAL_SECONDS="${PAPER_EVAL_VLLM_HEALTH_CHECK_INTERVAL_SECONDS:-5}"
VLLM_EXTRA_JSON="$(jq -cn \
  --arg max_num_seqs "$VLLM_MAX_NUM_SEQS" \
  --arg max_num_batched_tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
  --arg model_revision "$VLLM_MODEL_REVISION" \
  --arg eval_model_id "$EVAL_MODEL_ID" \
  --arg default_chat_template_kwargs "$VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS" \
  --arg enable_lora "$ENABLE_LORA" '
    {
      max_num_seqs: (if $max_num_seqs != "" then ($max_num_seqs | tonumber) else null end),
      max_num_batched_tokens: (
        if $max_num_batched_tokens != "" then ($max_num_batched_tokens | tonumber) else null end
      ),
      revision: (if $model_revision != "" then $model_revision else null end),
      served_model_name: (
        if $enable_lora != "1" and $eval_model_id != "" then [$eval_model_id] else null end
      ),
      default_chat_template_kwargs: (
        if $default_chat_template_kwargs != "" then ($default_chat_template_kwargs | fromjson) else null end
      )
    }
    | with_entries(select(.value != null))
  ')"

VLLM_CMD=(
  "${UV_RUN[@]}" python -m prime_rl.inference.server
  --model.name "$MODEL_REF"
  --server.port "$VLLM_PORT"
  --model.max-model-len "$VLLM_MAX_MODEL_LEN"
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --max-lora-rank "$VLLM_MAX_LORA_RANK"
  --api-server-count 1
)
if [ -n "$VLLM_TOOL_CALL_PARSER" ]; then
  VLLM_CMD+=(--model.tool-call-parser "$VLLM_TOOL_CALL_PARSER")
fi
if [ "$VLLM_EXTRA_JSON" != "{}" ]; then
  VLLM_CMD+=(--vllm-extra "$VLLM_EXTRA_JSON")
fi
if [ "$VLLM_ENFORCE_EAGER" = "1" ]; then
  VLLM_CMD+=(--model.enforce-eager)
fi
if [ -n "$VLLM_REASONING_PARSER" ]; then
  VLLM_CMD+=(--model.reasoning-parser "$VLLM_REASONING_PARSER")
fi
if [ -n "$VLLM_TP" ]; then
  VLLM_CMD+=(--parallel.tp "$VLLM_TP")
fi
if [ -n "$VLLM_DP" ]; then
  VLLM_CMD+=(--parallel.dp "$VLLM_DP")
fi
if [ -n "$DEPLOYMENT_GPUS_PER_NODE" ]; then
  VLLM_CMD+=(--deployment.gpus-per-node "$DEPLOYMENT_GPUS_PER_NODE")
fi
if [ "$ENABLE_LORA" = "1" ]; then
  VLLM_CMD+=(--enable-lora)
fi

echo "[INFO] Starting vLLM for $MODEL_TAG from $MODEL_REF on port $VLLM_PORT"
echo "[INFO] tool_call_parser=$VLLM_TOOL_CALL_PARSER reasoning_parser=${VLLM_REASONING_PARSER:-<none>}"
echo "[INFO] default_chat_template_kwargs=${VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS:-<none>}"
echo "[INFO] vllm enforce_eager=$VLLM_ENFORCE_EAGER"
echo "[INFO] vllm max_num_seqs=$VLLM_MAX_NUM_SEQS max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS:-default}"
if [ "$VLLM_EXTRA_JSON" != "{}" ]; then
  echo "[INFO] vllm_extra=$VLLM_EXTRA_JSON"
fi
echo "[INFO] model_id_for_eval=$EVAL_MODEL_ID"
echo "[INFO] vllm tp=${VLLM_TP:-1} dp=${VLLM_DP:-1} deployment_gpus_per_node=${DEPLOYMENT_GPUS_PER_NODE:-auto}"
if [ "$ENABLE_LORA" = "1" ]; then
  echo "[INFO] LoRA enabled: name=$LORA_MODEL_NAME path=$LORA_ADAPTER_DIR max_lora_rank=$VLLM_MAX_LORA_RANK"
fi
"${VLLM_CMD[@]}" &
VLLM_PID=$!

cleanup() {
  if kill -0 "$VLLM_PID" >/dev/null 2>&1; then
    kill "$VLLM_PID" >/dev/null 2>&1 || true
    wait "$VLLM_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "[INFO] Waiting for vLLM health: $VLLM_HEALTH_URL"
echo "[INFO] vLLM health attempts=$VLLM_HEALTH_CHECK_ATTEMPTS interval_seconds=$VLLM_HEALTH_CHECK_INTERVAL_SECONDS"
for _ in $(seq 1 "$VLLM_HEALTH_CHECK_ATTEMPTS"); do
  if curl -sf "$VLLM_HEALTH_URL" >/dev/null; then
    echo "[INFO] vLLM healthy"
    break
  fi
  sleep "$VLLM_HEALTH_CHECK_INTERVAL_SECONDS"
done

if ! curl -sf "$VLLM_HEALTH_URL" >/dev/null; then
  echo "[ERROR] vLLM failed to start" >&2
  exit 1
fi

if [ "$ENABLE_LORA" = "1" ]; then
  echo "[INFO] Loading LoRA adapter into running vLLM server"
  python3 - <<PY
import json
import urllib.request

url = "${VLLM_ENDPOINT}/load_lora_adapter"
payload = {
    "lora_name": "${LORA_MODEL_NAME}",
    "lora_path": "${LORA_ADAPTER_DIR}",
}
req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as resp:
    body = resp.read().decode("utf-8")
    print(f"[INFO] load_lora_adapter status={resp.status} body={body}")
PY
fi

echo "[INFO] Running paper static benchmark eval using experiment=$EVAL_EXPERIMENT"
RUNNER_ARGS=(
  +experiment="$EVAL_EXPERIMENT"
  agent.model="$EVAL_MODEL_ID"
  runner.output_dir="$EVAL_OUTPUT_DIR"
)
if [ -n "$EVAL_RUNNER_WORKERS" ]; then
  RUNNER_ARGS+=(runner.workers="$EVAL_RUNNER_WORKERS")
fi
if [ -n "$EVAL_RUNNER_RUNS_PER_ITEM" ]; then
  RUNNER_ARGS+=(runner.runs_per_item="$EVAL_RUNNER_RUNS_PER_ITEM")
fi
if [ -n "$EVAL_RUNNER_MAX_RETRIES_PER_RUN" ]; then
  RUNNER_ARGS+=(runner.max_retries_per_run="$EVAL_RUNNER_MAX_RETRIES_PER_RUN")
fi
if [ -n "$EVAL_RUNNER_MAX_RUNS" ]; then
  RUNNER_ARGS+=(runner.max_runs="$EVAL_RUNNER_MAX_RUNS")
fi
if [ -n "$EVAL_RUNNER_RUN_WALL_CLOCK_TIMEOUT" ]; then
  RUNNER_ARGS+=(runner.run_wall_clock_timeout="$EVAL_RUNNER_RUN_WALL_CLOCK_TIMEOUT")
fi
if [ -n "$EVAL_AGENT_MAX_TOKENS" ]; then
  RUNNER_ARGS+=(agent.params.max_tokens="$EVAL_AGENT_MAX_TOKENS")
fi
if [ -n "$EVAL_SCENARIO_MAX_PARALLEL_TOOL_CALLS" ]; then
  RUNNER_ARGS+=(scenario.max_parallel_tool_calls="$EVAL_SCENARIO_MAX_PARALLEL_TOOL_CALLS")
fi
if [ -n "$PAPER_EVAL_HYDRA_OVERRIDES" ]; then
  read -r -a HYDRA_EXTRA_ARGS <<<"$PAPER_EVAL_HYDRA_OVERRIDES"
  RUNNER_ARGS+=("${HYDRA_EXTRA_ARGS[@]}")
fi
OPENAI_API_BASE="$VLLM_API_BASE" OPENAI_API_KEY="default" \
  "${UV_RUN[@]}" python -m src.runner \
  "${RUNNER_ARGS[@]}"

TRACE_DIR="$EVAL_OUTPUT_DIR/traces/$EVAL_MODEL_ID"
if [ ! -d "$TRACE_DIR" ]; then
  TRACE_DIR="$EVAL_OUTPUT_DIR/traces/$MODEL_TAG"
fi
if [ ! -d "$TRACE_DIR" ]; then
  TRACE_DIR="$(${UV_RUN[@]} python - "$EVAL_OUTPUT_DIR" "$EVAL_MODEL_ID" <<'PY'
import sys
from src.paths import traces_dir
print(traces_dir(sys.argv[1], sys.argv[2]))
PY
)"
fi
if [ ! -d "$TRACE_DIR" ]; then
  echo "[ERROR] expected trace dir not found: $TRACE_DIR" >&2
  exit 1
fi

echo "[INFO] Running evaluation analysis for $EVAL_OUTPUT_DIR"
ANALYZE_ARGS=(--base-dir "$EVAL_OUTPUT_DIR" --experiment "$EVAL_EXPERIMENT" --all-models)
if [ "$EVAL_ANALYZE_ALLOW_INCOMPLETE_RUNS" = "1" ]; then
  ANALYZE_ARGS+=(--allow-incomplete-runs)
fi
if [ -n "$EVAL_ANALYZE_EXPECTED_RUNS_PER_SCENARIO" ]; then
  ANALYZE_ARGS+=(--expected-runs-per-scenario "$EVAL_ANALYZE_EXPECTED_RUNS_PER_SCENARIO")
fi
"${UV_RUN[@]}" python -m src.evaluation.analyze "${ANALYZE_ARGS[@]}"

PROVENANCE_FILE="$EVAL_OUTPUT_DIR/provenance_model_eval.json"
PROJECT_REALPATH="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$PROJECT_DIR")"
GIT_SHA="$(git -C "$PROJECT_DIR" rev-parse HEAD)"
UTC_NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
MODEL_REF_REALPATH=""
if [ -d "$MODEL_REF" ]; then
  MODEL_REF_REALPATH="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$MODEL_REF")"
fi

export MODEL_REF MODEL_REF_REALPATH MODEL_TAG EVAL_EXPERIMENT EVAL_OUTPUT_DIR VLLM_API_BASE
export VLLM_MODEL_REVISION VLLM_MAX_MODEL_LEN
export VLLM_GPU_MEMORY_UTILIZATION VLLM_MAX_NUM_SEQS VLLM_MAX_NUM_BATCHED_TOKENS
export VLLM_TOOL_CALL_PARSER VLLM_REASONING_PARSER VLLM_ENFORCE_EAGER
export VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS
export VLLM_TP VLLM_DP DEPLOYMENT_GPUS_PER_NODE
export UV_GROUP UV_WITH PATCH_VLLM_GEMMA4_BNB
export ENABLE_LORA LORA_ADAPTER_DIR LORA_MODEL_NAME VLLM_MAX_LORA_RANK EVAL_MODEL_ID
export PROJECT_REALPATH GIT_SHA UTC_NOW
export PAPER_EVAL_HYDRA_OVERRIDES

python3 - "$PROVENANCE_FILE" <<'PY'
import json
import os
import sys

from src.paths import path_from_outputs_or_none

output_path = sys.argv[1]
payload = {
    "created_at_utc": os.environ["UTC_NOW"],
    "project_dir": os.environ["PROJECT_REALPATH"],
    "git_sha": os.environ["GIT_SHA"],
    "model_ref": os.environ["MODEL_REF"],
    "model_ref_realpath": os.environ.get("MODEL_REF_REALPATH") or None,
    "model_revision": os.environ.get("VLLM_MODEL_REVISION") or None,
    "model_tag": os.environ["MODEL_TAG"],
    "eval_model_id": os.environ["EVAL_MODEL_ID"],
    "eval_experiment": os.environ["EVAL_EXPERIMENT"],
    "eval_hydra_overrides": os.environ.get("PAPER_EVAL_HYDRA_OVERRIDES") or None,
    "eval_output_dir": os.environ["EVAL_OUTPUT_DIR"],
    "eval_output_dir_path_from_outputs": None,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "openai_api_base": os.environ["VLLM_API_BASE"],
    "uv_group": os.environ.get("UV_GROUP") or None,
    "uv_with": os.environ.get("UV_WITH") or None,
    "uv_project_environment": os.environ.get("UV_PROJECT_ENVIRONMENT") or None,
    "patch_vllm_gemma4_bnb": os.environ.get("PATCH_VLLM_GEMMA4_BNB") == "1",
    "vllm": {
        "max_model_len": os.environ["VLLM_MAX_MODEL_LEN"],
        "gpu_memory_utilization": os.environ["VLLM_GPU_MEMORY_UTILIZATION"],
        "max_num_seqs": int(os.environ["VLLM_MAX_NUM_SEQS"]),
        "max_num_batched_tokens": int(os.environ["VLLM_MAX_NUM_BATCHED_TOKENS"]) if os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS") else None,
        "tool_call_parser": os.environ["VLLM_TOOL_CALL_PARSER"],
        "reasoning_parser": os.environ.get("VLLM_REASONING_PARSER") or None,
        "default_chat_template_kwargs": json.loads(os.environ["VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS"]) if os.environ.get("VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS") else None,
        "enforce_eager": os.environ.get("VLLM_ENFORCE_EAGER") == "1",
        "tensor_parallel_size": int(os.environ["VLLM_TP"]) if os.environ.get("VLLM_TP") else 1,
        "data_parallel_size": int(os.environ["VLLM_DP"]) if os.environ.get("VLLM_DP") else 1,
        "deployment_gpus_per_node": int(os.environ["DEPLOYMENT_GPUS_PER_NODE"]) if os.environ.get("DEPLOYMENT_GPUS_PER_NODE") else None,
        "enable_lora": os.environ["ENABLE_LORA"] == "1",
        "lora_model_name": os.environ.get("LORA_MODEL_NAME") or None,
        "lora_adapter_dir": os.environ.get("LORA_ADAPTER_DIR") or None,
        "max_lora_rank": os.environ.get("VLLM_MAX_LORA_RANK") or None,
    },
}

payload["eval_output_dir_path_from_outputs"] = path_from_outputs_or_none(os.environ["EVAL_OUTPUT_DIR"])
payload["vllm"]["lora_adapter_dir_path_from_outputs"] = path_from_outputs_or_none(
    os.environ.get("LORA_ADAPTER_DIR")
)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
PY

echo "[INFO] Wrote provenance: $PROVENANCE_FILE"
echo "[INFO] Paper static benchmark evaluation finished for $MODEL_TAG"
