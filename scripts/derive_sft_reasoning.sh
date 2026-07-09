#!/usr/bin/env bash
# Derive SFT reasoning variants from passing long_reasoning traces, then assemble them.
# Usage: ./scripts/derive_sft_reasoning.sh [guided|unguided] [no_reasoning|short_reasoning] [training|validation|full] [teacher model] [extra SFT Hydra overrides...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  echo "Usage: $0 [guided|unguided] [no_reasoning|short_reasoning] [training|validation|full] [teacher model] [extra SFT Hydra overrides...]"
}

REGIME="${SFT_REGIME:-unguided}"
if (( $# > 0 )); then
  case "$1" in
    guided|unguided)
      REGIME="$1"
      shift
      ;;
  esac
fi
case "$REGIME" in
  guided|unguided) ;;
  *)
    usage
    exit 1
    ;;
esac

TEACHER="deepseek"

VARIANT="no_reasoning"
if (( $# > 0 )); then
  case "$1" in
    no_reasoning|short_reasoning)
      VARIANT="$1"
      shift
      ;;
    training|validation|full) ;;
    *)
      echo "Unsupported variant: $1"
      exit 1
      ;;
  esac
fi

MODE="full"
if (( $# > 0 )); then
  case "$1" in
    training|validation|full)
      MODE="$1"
      shift
      ;;
  esac
fi
case "$MODE" in
  training|validation|full) ;;
  *)
    usage
    exit 1
    ;;
esac

TEACHER_MODEL="${SFT_TEACHER_MODEL:-${DEFAULT_MODEL:-}}"
if [[ -z "$TEACHER_MODEL" && $# -gt 0 ]]; then
  case "$1" in
    *=*|+*=*) ;;
    deepseek/*)
      TEACHER_MODEL="$1"
      shift
      ;;
  esac
fi

TEACHER_ARGS=("datasets/sft/teacher=deepseek")

run_one() {
  local split="$1"
  shift
  local derive_experiment="derive/standard_${REGIME}_deepseek_${VARIANT}_${split}"
  local assembly_experiment="assembly/standard_${REGIME}_deepseek_${split}"

  local derive_cmd=(uv run python -m src.dataset.privesc.sft_reasoning_variants \
    +experiment="$derive_experiment" \
    "${TEACHER_ARGS[@]}")
  if [[ -n "$TEACHER_MODEL" ]]; then
    derive_cmd+=(agent.model="$TEACHER_MODEL")
    derive_cmd+=(datasets.sft.teacher_model="$TEACHER_MODEL")
  fi
  derive_cmd+=(
    reasoning_variant_derivation.incremental=$([[ "$VARIANT" == "short_reasoning" ]] && echo true || echo false)
    "$@"
  )
  "${derive_cmd[@]}"

  if [[ "$VARIANT" == "no_reasoning" ]]; then
    local quality_overrides=(
      datasets.sft.quality.reject_on_empty_reasoning=false
      datasets.sft.quality.min_reasoning_length=0
    )
  else
    local quality_overrides=(datasets.sft.quality.min_reasoning_length=1)
  fi

  local assembly_cmd=(uv run python -m src.dataset.privesc.sft \
    +experiment="$assembly_experiment" \
    "${TEACHER_ARGS[@]}")
  if [[ -n "$TEACHER_MODEL" ]]; then
    assembly_cmd+=(datasets.sft.teacher_model="$TEACHER_MODEL")
  fi
  assembly_cmd+=(
    datasets.sft.reasoning_variant="$VARIANT"
    "${quality_overrides[@]}"
    datasets.sft.prune=false
    "$@"
  )
  "${assembly_cmd[@]}"
}

if [[ "$MODE" == "full" ]]; then
  run_one training "$@"
  run_one validation "$@"
else
  run_one "$MODE" "$@"
fi
