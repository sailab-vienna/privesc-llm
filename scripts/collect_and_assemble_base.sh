#!/usr/bin/env bash
# Core collect-and-assemble logic. Not intended to be called directly.
# Use collect_and_assemble.sh instead.
#
# Positional arguments:
#   1: mode   (training|validation|full)
#   2: teacher model  (optional; defaults to DEFAULT_MODEL)
#   3+: extra Hydra overrides. Collection-only overrides such as runner.*,
#       agent.*, and prompts.* are not forwarded to SFT assembly.
#
# Required environment variables are set by the caller wrapper.
set -euo pipefail

: "${TRACE_ROOT:?TRACE_ROOT must be set by caller}"
: "${TRACE_EXPERIMENT_TRAINING:?TRACE_EXPERIMENT_TRAINING must be set by caller}"
: "${TRACE_EXPERIMENT_VALIDATION:?TRACE_EXPERIMENT_VALIDATION must be set by caller}"
: "${TRACE_GENERATORS_TRAINING:?TRACE_GENERATORS_TRAINING must be set by caller}"
: "${TRACE_GENERATORS_VALIDATION:?TRACE_GENERATORS_VALIDATION must be set by caller}"
: "${TRACE_SOURCE_TRAINING:?TRACE_SOURCE_TRAINING must be set by caller}"
: "${TRACE_SOURCE_VALIDATION:?TRACE_SOURCE_VALIDATION must be set by caller}"
: "${DATASET_CONFIG_TRAINING:?DATASET_CONFIG_TRAINING must be set by caller}"
: "${DATASET_CONFIG_VALIDATION:?DATASET_CONFIG_VALIDATION must be set by caller}"
: "${DATASET_PROFILE:?DATASET_PROFILE must be set by caller}"
: "${DATASET_REGIME:?DATASET_REGIME must be set by caller}"
: "${DATASET_TEACHER:?DATASET_TEACHER must be set by caller}"
: "${DATASET_REASONING_VARIANT:?DATASET_REASONING_VARIANT must be set by caller}"
: "${TRAINING_SEED:?TRAINING_SEED must be set by caller}"
: "${VALIDATION_SEED:?VALIDATION_SEED must be set by caller}"

DEFAULT_MODEL="${DEFAULT_MODEL:-deepseek/deepseek-v4-flash}"
MODE="${1:-full}"
MODEL_FULL="${2:-$DEFAULT_MODEL}"
MODEL_DIR_RAW="${MODEL_FULL##*/}"
MODEL_DIR="${MODEL_DIR_RAW%%:*}"
EXTRA_HYDRA_ARGS=()
if (( $# > 2 )); then
  EXTRA_HYDRA_ARGS=("${@:3}")
fi
MAX_ITERS="${MAX_ITERS:-200}"

SFT_HYDRA_ARGS=()
for ((i = 0; i < ${#EXTRA_HYDRA_ARGS[@]}; i++)); do
  arg="${EXTRA_HYDRA_ARGS[$i]}"
  case "$arg" in
    runner.output_dir=*|+runner.output_dir=*)
      echo "Error: do not pass runner.output_dir to this wrapper; set TRACE_ROOT instead."
      exit 1
      ;;
    runner.*|+runner.*|agent.*|+agent.*|prompts.*|+prompts.*) ;;
    *)
      SFT_HYDRA_ARGS+=("$arg")
      ;;
  esac
done

run_python_module() {
  local module="$1"
  shift
  local cmd=(uv run python -m "$module" "$@")
  if (( ${#EXTRA_HYDRA_ARGS[@]} > 0 )); then
    cmd+=("${EXTRA_HYDRA_ARGS[@]}")
  fi
  "${cmd[@]}"
}

if [[ "$MODE" != "training" && "$MODE" != "validation" && "$MODE" != "full" ]]; then
  echo "Usage: $0 [training|validation|full] [teacher model]"
  echo "  full:                  Collect training, then validation"
  echo "  training:              Collect training traces"
  echo "  validation:            Collect validation traces"
  echo "  teacher:   Teacher model used for output paths (default: ${DEFAULT_MODEL})"
  echo "  overrides:  Extra Hydra overrides; runner/agent/prompt overrides are collection-only"
  exit 1
fi

if [[ "$MODE" == "full" ]]; then
  echo "Mode full: running training then validation"
  if (( ${#EXTRA_HYDRA_ARGS[@]} > 0 )); then
    "$0" training "$MODEL_FULL" "${EXTRA_HYDRA_ARGS[@]}"
    "$0" validation "$MODEL_FULL" "${EXTRA_HYDRA_ARGS[@]}"
  else
    "$0" training "$MODEL_FULL"
    "$0" validation "$MODEL_FULL"
  fi
  exit 0
fi

if [[ "$MODE" == "training" ]]; then
  TRACE_EXPERIMENT="$TRACE_EXPERIMENT_TRAINING"
  TRACE_GENERATORS_PROFILE="$TRACE_GENERATORS_TRAINING"
  TRACE_SOURCE_DIR="$TRACE_SOURCE_TRAINING"
  DATASET_CONFIG="$DATASET_CONFIG_TRAINING"
else
  TRACE_EXPERIMENT="$TRACE_EXPERIMENT_VALIDATION"
  TRACE_GENERATORS_PROFILE="$TRACE_GENERATORS_VALIDATION"
  TRACE_SOURCE_DIR="$TRACE_SOURCE_VALIDATION"
  DATASET_CONFIG="$DATASET_CONFIG_VALIDATION"
fi

RUNNER_OUTPUT_DIR="${TRACE_ROOT%/}/${TRACE_SOURCE_DIR}"
TRACES_DIR="${RUNNER_OUTPUT_DIR}/traces/${MODEL_DIR}"
SUMMARY_PATH="${RUNNER_OUTPUT_DIR}/stats/collection_summary.json"

hr() {
  printf '%*s\n' 72 '' | tr ' ' '-'
}

print_config() {
  hr
  printf '%-14s %s\n' "Mode" "$MODE"
  printf '%-14s %s\n' "Teacher" "$MODEL_FULL"
  printf '%-14s %s\n' "Experiment" "$TRACE_EXPERIMENT"
  printf '%-14s %s\n' "Generators" "$TRACE_GENERATORS_PROFILE"
  printf '%-14s %s\n' "Traces" "$TRACES_DIR"
  printf '%-14s %s\n' "Dataset cfg" "$DATASET_CONFIG"
  printf '%-14s %s\n' "Summary" "$SUMMARY_PATH"
  printf '%-14s %s\n' "Prune" "false"
  hr
}

generator_count() {
  uv run python -m src.dataset.privesc.collection_stats \
    generators "$SUMMARY_PATH" | sed '/^$/d' | wc -l | tr -d ' '
}

target_per_generator() {
  uv run python -m src.dataset.privesc.collection_stats target "$SUMMARY_PATH"
}

preflight_seed_collisions() {
  local train_seed="$TRAINING_SEED"
  local val_seed="$VALIDATION_SEED"
  local runs
  local num_gens
  runs=$(target_per_generator)
  num_gens=$(generator_count)
  if (( num_gens == 0 )); then
    echo "Error: no generators found in collection summary"
    exit 5
  fi

  local span=$(( runs * num_gens ))
  local train_max=$(( train_seed + span - 1 ))

  echo "Seed collision preflight"
  printf '  %-10s %d runs/generator x %d generators\n' \
    "span" "$runs" "$num_gens"
  printf '  %-10s training starts at %d, validation starts at %d\n' \
    "seeds" "$train_seed" "$val_seed"

  if (( train_max >= val_seed )); then
    echo "Error: training/validation seed ranges overlap"
    echo "Fix the wrapper seed values or the matching runner config."
    exit 4
  fi
  echo "Seed ranges do not collide for current split span"
}

check_all_complete() {
  (( $(missing_total) == 0 ))
}

missing_total() {
  uv run python -m src.dataset.privesc.collection_stats \
    missing "$SUMMARY_PATH"
}

print_status() {
  uv run python -m src.dataset.privesc.collection_stats \
    status "$SUMMARY_PATH"
}

run_collection_stats() {
  run_python_module src.dataset.privesc.collection_stats \
    +experiment="$TRACE_EXPERIMENT" \
    generators="$TRACE_GENERATORS_PROFILE" \
    agent.model="$MODEL_FULL" \
    datasets.sft.profile="$DATASET_PROFILE" \
    datasets.sft.regime="$DATASET_REGIME" \
    datasets.sft.teacher="$DATASET_TEACHER" \
    datasets.sft.reasoning_variant="$DATASET_REASONING_VARIANT" \
    datasets.sft.teacher_model="$MODEL_FULL" \
    datasets.sft.trace_root="$TRACE_ROOT" \
    datasets.sft.source_dir="$TRACE_SOURCE_DIR"
}

run_sft_assembly() {
  local target
  target=$(target_per_generator)
  local cmd=(uv run python -m src.dataset.privesc.sft \
    "runner=trace_collection/standard_${TRACE_SOURCE_DIR}" \
    "generators=${TRACE_GENERATORS_PROFILE}" \
    "datasets/sft=${DATASET_CONFIG}" \
    "datasets/sft/quality=${DATASET_REGIME}" \
    datasets.sft.profile="$DATASET_PROFILE" \
    datasets.sft.regime="$DATASET_REGIME" \
    datasets.sft.teacher="$DATASET_TEACHER" \
    datasets.sft.reasoning_variant="$DATASET_REASONING_VARIANT" \
    datasets.sft.teacher_model="$MODEL_FULL" \
    datasets.sft.trace_root="$TRACE_ROOT" \
    datasets.sft.source_dir="$TRACE_SOURCE_DIR" \
    datasets.sft.max_per_generator="$target" \
    datasets.sft.prune=false)
  if (( ${#SFT_HYDRA_ARGS[@]} > 0 )); then
    cmd+=("${SFT_HYDRA_ARGS[@]}")
  fi
  "${cmd[@]}"
}

iteration=0
prev_missing=-1
print_config

echo "Initial collection summary"
run_collection_stats
preflight_seed_collisions

while ! check_all_complete; do
  ((++iteration))
  if (( iteration > MAX_ITERS )); then
    echo "Error: exceeded MAX_ITERS=$MAX_ITERS without reaching targets"
    print_status
    exit 2
  fi
  m=$(missing_total)
  echo "Iteration ${iteration}/${MAX_ITERS} (missing: ${m})"

  run_python_module src.runner \
    +experiment="$TRACE_EXPERIMENT" \
    generators="$TRACE_GENERATORS_PROFILE" \
    runner.output_dir="$RUNNER_OUTPUT_DIR" \
    agent.model="$MODEL_FULL"

  run_collection_stats

  m=$(missing_total)
  if (( iteration == 1 || m == 0 || iteration % 5 == 0 || m != prev_missing )); then
    print_status
  else
    echo "Progress: missing ${m}"
  fi
  prev_missing=$m
done

echo "Done: reached $(target_per_generator)/generator"
echo "Assembling dataset"
run_sft_assembly

print_status
