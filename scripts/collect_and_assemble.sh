#!/usr/bin/env bash
# Collect and assemble SFT training/validation traces.
# Usage: ./scripts/collect_and_assemble.sh [guided|unguided] deepseek [training|validation|full] [teacher model] [extra hydra overrides...]
# Runner/agent/prompt overrides are collection-only; dataset overrides also reach assembly.
set -euo pipefail

usage() {
  echo "Usage: $0 [guided|unguided] deepseek [training|validation|full] [teacher model] [extra hydra overrides...]"
}

if (( $# < 1 )); then
  usage
  exit 1
fi

REGIME="${SFT_REGIME:-unguided}"
case "$1" in
  guided|unguided)
    REGIME="$1"
    shift
    ;;
esac

if (( $# < 1 )); then
  usage
  exit 1
fi

TEACHER="$1"
shift
case "$TEACHER" in
  deepseek)
    export DEFAULT_MODEL="${DEFAULT_MODEL:-deepseek/deepseek-v4-flash}"
    ;;
  *)
    usage
    exit 1
    ;;
esac

export SFT_REASONING_VARIANT="${SFT_REASONING_VARIANT:-long_reasoning}"
export TRACE_ROOT="${TRACE_ROOT:-outputs/traces/trace_collection/standard/${REGIME}/${TEACHER}/${SFT_REASONING_VARIANT}}"
export TRACE_EXPERIMENT_TRAINING="trace/standard_${REGIME}_${TEACHER}_training"
export TRACE_EXPERIMENT_VALIDATION="trace/standard_${REGIME}_${TEACHER}_validation"
export TRACE_GENERATORS_TRAINING="training"
export TRACE_GENERATORS_VALIDATION="holdout"
export DATASET_CONFIG_TRAINING="collection/standard/training"
export DATASET_CONFIG_VALIDATION="collection/standard/validation"
export DATASET_PROFILE="standard"
export DATASET_REGIME="$REGIME"
export DATASET_TEACHER="$TEACHER"
export DATASET_REASONING_VARIANT="$SFT_REASONING_VARIANT"
export SFT_TRAIN_DATA_DIR="${SFT_TRAIN_DATA_DIR:-outputs/data/privesc_sft/standard/${REGIME}/${TEACHER}/${SFT_REASONING_VARIANT}/training}"
export SFT_VAL_DATA_DIR="${SFT_VAL_DATA_DIR:-outputs/data/privesc_sft/standard/${REGIME}/${TEACHER}/${SFT_REASONING_VARIANT}/validation}"

export TRACE_SOURCE_TRAINING="training"
export TRACE_SOURCE_VALIDATION="validation"
export TRAINING_SEED=42
export VALIDATION_SEED=10000000

exec "$(dirname "$0")/collect_and_assemble_base.sh" "$@"
