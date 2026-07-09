#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

SUBMODULE_NAME="prime-rl"
SUBMODULE_PATH="external/prime-rl"
SUBMODULE_DIR="$REPO_ROOT/$SUBMODULE_PATH"
PRIME_RL_DIR="${PRIME_RL_DIR:-$SUBMODULE_DIR}"

PATCH_FILES=(
  "$REPO_ROOT/src/rl/prime_rl/patches/rl-lora-init-adapter-prime-rl.patch"
  "$REPO_ROOT/src/rl/prime_rl/patches/rl-lm-head-temperature-compat-prime-rl.patch"
  "$REPO_ROOT/src/rl/prime_rl/patches/rl-metrics-all-aggregation-prime-rl.patch"
  "$REPO_ROOT/src/rl/prime_rl/patches/rl-buffer-round-robin-sampling-prime-rl.patch"
  "$REPO_ROOT/src/rl/prime_rl/patches/rl-rollout-task-state-column-prime-rl.patch"
  "$REPO_ROOT/src/rl/prime_rl/patches/rl-local-inference-ready-prime-rl.patch"
  "$REPO_ROOT/src/rl/prime_rl/patches/rl-vllm-dp-coordinator-timeout-prime-rl.patch"
)

if [ -n "${PATCH_FILE:-}" ]; then
  PATCH_FILES=("$PATCH_FILE")
fi

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

check_checkout() {
  local repo_dir="$1"
  local repo_label="$2"

  git -C "$repo_dir" rev-parse --git-dir >/dev/null 2>&1 || fail "$repo_label is not a git checkout: $repo_dir"
}

require_clean_checkout() {
  local repo_dir="$1"
  local repo_label="$2"

  [ -z "$(git -C "$repo_dir" status --short --untracked-files=all)" ] || \
    fail "$repo_label is not clean: $repo_dir"
}

normalize_patch_files() {
  local i
  local patch_file

  for i in "${!PATCH_FILES[@]}"; do
    patch_file="${PATCH_FILES[$i]}"
    if [[ "$patch_file" != /* ]]; then
      patch_file="$REPO_ROOT/$patch_file"
      PATCH_FILES[$i]="$patch_file"
    fi
    [ -f "$patch_file" ] || fail "Patch file does not exist: $patch_file"
  done
}

apply_patch_if_needed() {
  local repo_dir="$1"
  local patch_file="$2"

  if git -C "$repo_dir" apply --check -- "$patch_file"; then
    git -C "$repo_dir" apply -- "$patch_file"
  elif git -C "$repo_dir" apply --reverse --check -- "$patch_file"; then
    echo "[INFO] Prime-RL patch already applied: $patch_file"
  else
    git -C "$repo_dir" apply --check -- "$patch_file"
  fi
}

sync_submodule_checkout() {
  echo "[INFO] Syncing $SUBMODULE_PATH to $EXPECTED_SUBMODULE_COMMIT"
  git submodule update --init --checkout --force -- "$SUBMODULE_PATH" >/dev/null
  git -C "$SUBMODULE_DIR" clean -fd >/dev/null

  check_checkout "$SUBMODULE_DIR" "$SUBMODULE_PATH"
  [ "$(git -C "$SUBMODULE_DIR" remote get-url origin)" = "$EXPECTED_SUBMODULE_URL" ] || \
    fail "$SUBMODULE_PATH origin URL does not match .gitmodules"
  [ "$(git -C "$SUBMODULE_DIR" rev-parse HEAD)" = "$EXPECTED_SUBMODULE_COMMIT" ] || \
    fail "$SUBMODULE_PATH is not at the recorded submodule commit"
  require_clean_checkout "$SUBMODULE_DIR" "$SUBMODULE_PATH"
}

sync_checkout() {
  local repo_dir="$1"
  local repo_label="$2"

  check_checkout "$repo_dir" "$repo_label"
  if [ "$(git -C "$repo_dir" rev-parse HEAD)" != "$EXPECTED_SUBMODULE_COMMIT" ] || \
     [ -n "$(git -C "$repo_dir" status --short --untracked-files=all)" ]; then
    echo "[INFO] Syncing $repo_label to $EXPECTED_SUBMODULE_COMMIT"
    git -C "$repo_dir" reset --hard "$EXPECTED_SUBMODULE_COMMIT" >/dev/null
    git -C "$repo_dir" clean -fd >/dev/null
  fi

  require_clean_checkout "$repo_dir" "$repo_label"
}

EXPECTED_SUBMODULE_URL="$(git config -f "$REPO_ROOT/.gitmodules" --get submodule.$SUBMODULE_NAME.url || true)"
[ -n "$EXPECTED_SUBMODULE_URL" ] || fail "Missing .gitmodules entry for $SUBMODULE_NAME"

RECORDED_SUBMODULE_COMMIT="$(git rev-parse "HEAD:$SUBMODULE_PATH")"
[ -n "$RECORDED_SUBMODULE_COMMIT" ] || fail "Missing gitlink for $SUBMODULE_PATH at HEAD"

CURRENT_SUBMODULE_COMMIT="$(git -C "$SUBMODULE_DIR" rev-parse HEAD)"
EXPECTED_SUBMODULE_COMMIT="$RECORDED_SUBMODULE_COMMIT"
USE_CURRENT_SUBMODULE_COMMIT=0
if [ "$PRIME_RL_DIR" = "$SUBMODULE_DIR" ] && [ "$CURRENT_SUBMODULE_COMMIT" != "$RECORDED_SUBMODULE_COMMIT" ]; then
  EXPECTED_SUBMODULE_COMMIT="$CURRENT_SUBMODULE_COMMIT"
  USE_CURRENT_SUBMODULE_COMMIT=1
fi

normalize_patch_files

if [ "$PRIME_RL_DIR" = "$SUBMODULE_DIR" ]; then
  if [ "$USE_CURRENT_SUBMODULE_COMMIT" = "1" ]; then
    check_checkout "$SUBMODULE_DIR" "$SUBMODULE_PATH"
    [ "$(git -C "$SUBMODULE_DIR" remote get-url origin)" = "$EXPECTED_SUBMODULE_URL" ] || \
      fail "$SUBMODULE_PATH origin URL does not match .gitmodules"
    require_clean_checkout "$SUBMODULE_DIR" "$SUBMODULE_PATH"
  else
    sync_submodule_checkout
  fi
else
  sync_checkout "$PRIME_RL_DIR" 'PRIME_RL_DIR'
fi

PRECHECK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/prime-rl-patch-check.XXXXXX")"
cleanup() {
  git -C "$PRIME_RL_DIR" worktree remove --force "$PRECHECK_DIR" >/dev/null 2>&1 || true
  rm -rf "$PRECHECK_DIR"
}
trap cleanup EXIT

echo "[INFO] Verifying $SUBMODULE_PATH matches the submodule definition at $EXPECTED_SUBMODULE_COMMIT"
echo "[INFO] Preflighting Prime-RL patches in temporary worktree: $PRECHECK_DIR"

git -C "$PRIME_RL_DIR" worktree add --detach "$PRECHECK_DIR" "$EXPECTED_SUBMODULE_COMMIT" >/dev/null

for patch_file in "${PATCH_FILES[@]}"; do
  echo "[INFO] Checking Prime-RL patch: $patch_file"
  apply_patch_if_needed "$PRECHECK_DIR" "$patch_file"
done

for patch_file in "${PATCH_FILES[@]}"; do
  echo "[INFO] Applying Prime-RL patch: $patch_file"
  apply_patch_if_needed "$PRIME_RL_DIR" "$patch_file"
done

echo "[INFO] Applied or verified ${#PATCH_FILES[@]} Prime-RL patch(es) cleanly in $PRIME_RL_DIR"
