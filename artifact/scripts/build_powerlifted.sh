#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
SOURCE_DIR="${POWERLIFTED_SOURCE_DIR:-$REPO_ROOT/external/.powerlifted-src}"
INSTALL_DIR="${POWERLIFTED_INSTALL_DIR:-$REPO_ROOT/external/powerlifted}"
POWERLIFTED_REV="${POWERLIFTED_REV:-2bd3bc6ca09d2b239e7699e8bb164703b6153b7d}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
BOOST_PREFIX="${BOOST_PREFIX:-}"

if ! command -v git >/dev/null 2>&1; then
  echo "[ERROR] Missing required command: git" >&2
  exit 1
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "[ERROR] Missing required command: cmake" >&2
  exit 1
fi
if ! command -v make >/dev/null 2>&1; then
  echo "[ERROR] Missing required command: make" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] Missing required command: python3" >&2
  exit 1
fi
if ! command -v gcc >/dev/null 2>&1 || ! command -v g++ >/dev/null 2>&1; then
  echo "[ERROR] Missing required compilers: gcc and g++" >&2
  exit 1
fi

if [ -z "$BOOST_PREFIX" ] && command -v brew >/dev/null 2>&1; then
  if brew --prefix boost >/dev/null 2>&1; then
    BOOST_PREFIX="$(brew --prefix boost)"
  fi
fi
if [ -z "$BOOST_PREFIX" ]; then
  echo "[ERROR] Could not resolve Boost. Install Homebrew boost or set BOOST_PREFIX." >&2
  exit 1
fi
if [ ! -d "$BOOST_PREFIX" ]; then
  echo "[ERROR] BOOST_PREFIX does not exist: $BOOST_PREFIX" >&2
  exit 1
fi

mkdir -p "$(dirname -- "$SOURCE_DIR")"
if [ ! -d "$SOURCE_DIR/.git" ]; then
  git clone https://github.com/peperunas/planner8.git "$SOURCE_DIR"
fi

git -C "$SOURCE_DIR" fetch --depth 1 origin "$POWERLIFTED_REV"
git -C "$SOURCE_DIR" checkout --detach "$POWERLIFTED_REV"
git -C "$SOURCE_DIR" clean -fdx

python3 - "$SOURCE_DIR/ext/powerlifted/build.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = "from distutils.dir_util import copy_tree\nfrom shutil import copytree\n"
new = "from shutil import copytree\n\n\ndef copy_tree(src, dst):\n    copytree(src, dst, dirs_exist_ok=True)\n"
if old not in text:
    raise SystemExit(f"Unexpected build.py contents: {path}")
path.write_text(text.replace(old, new))
PY

python3 - "$SOURCE_DIR/ext/powerlifted/src/search/CMakeLists.txt" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old_flags = 'set(CMAKE_CXX_FLAGS "-Wall -Wnon-virtual-dtor")'
new_flags = 'set(CMAKE_CXX_FLAGS "-Wall -Wnon-virtual-dtor -Wno-deprecated-declarations -Wno-deprecated-builtins")'
old_release = '        "-O3 -g -DNDEBUG -fomit-frame-pointer -Werror")'
new_release = '        "-O3 -g -DNDEBUG -fomit-frame-pointer")'
if old_flags not in text or old_release not in text:
    raise SystemExit(f"Unexpected search CMakeLists contents: {path}")
text = text.replace(old_flags, new_flags)
text = text.replace(old_release, new_release)
path.write_text(text)
PY

pushd "$SOURCE_DIR/ext/cpddl" >/dev/null
make -j"$JOBS"
make -C bin
popd >/dev/null

POWERLIFTED_SRC="$SOURCE_DIR/ext/powerlifted"
rm -rf "$POWERLIFTED_SRC/builds/release"
mkdir -p "$POWERLIFTED_SRC/builds/release/search"
cp -R "$POWERLIFTED_SRC/src/translator" "$POWERLIFTED_SRC/builds/release/translator"
python3 - "$POWERLIFTED_SRC/builds/release/translator/translate.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
lines = path.read_text().splitlines()
if lines and lines[0].startswith('#!'):
    lines[0] = '#!/usr/bin/env python3'
else:
    lines.insert(0, '#!/usr/bin/env python3')
path.write_text('\n'.join(lines) + '\n')
path.chmod(0o755)
PY

pushd "$POWERLIFTED_SRC/builds/release/search" >/dev/null
cmake "$POWERLIFTED_SRC/src/search" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$BOOST_PREFIX"
make -j"$JOBS"
popd >/dev/null

rm -rf "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR/bin" "$INSTALL_DIR/libexec/powerlifted/builds" "$INSTALL_DIR/libexec/cpddl/bin"
cp "$POWERLIFTED_SRC/powerlifted.py" "$POWERLIFTED_SRC/build.py" "$INSTALL_DIR/libexec/powerlifted/"
cp -R "$POWERLIFTED_SRC/driver" "$INSTALL_DIR/libexec/powerlifted/"
cp -R "$POWERLIFTED_SRC/builds/release" "$INSTALL_DIR/libexec/powerlifted/builds/"
cp "$SOURCE_DIR/ext/cpddl/bin/"pddl* "$INSTALL_DIR/libexec/cpddl/bin/"
cat > "$INSTALL_DIR/bin/powerlifted" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export CPDDL_BIN="$ROOT/libexec/cpddl/bin"
exec python3 "$ROOT/libexec/powerlifted/powerlifted.py" "$@"
EOF
chmod +x "$INSTALL_DIR/bin/powerlifted"

"$INSTALL_DIR/bin/powerlifted" --help >/dev/null

echo "[OK] Installed PowerLifted to $INSTALL_DIR/bin/powerlifted"
