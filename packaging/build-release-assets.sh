#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <tag> <platform>" >&2
  exit 2
fi

TAG="$1"
PLATFORM="$2"

if [[ ! "$TAG" =~ ^v[0-9]+[.][0-9]+[.][0-9]+([-+][A-Za-z0-9.-]+)?$ ]]; then
  echo "Tag must look like v2.0.2: $TAG" >&2
  exit 2
fi

ROOT="$(git rev-parse --show-toplevel)"
RELEASE_DIR="$ROOT/dist/release"
WORK_DIR="$RELEASE_DIR/work"
PKG_NAME="llmux-${TAG}-${PLATFORM}"
PKG_PARENT="$WORK_DIR/${PLATFORM}"
PKG_DIR="$PKG_PARENT/$PKG_NAME"

rm -rf "$PKG_PARENT"
mkdir -p "$PKG_PARENT" "$RELEASE_DIR"

git -C "$ROOT" archive --format=tar --prefix="$PKG_NAME/" HEAD | tar -xf - -C "$PKG_PARENT"

cat > "$PKG_DIR/install.sh" <<'INSTALL_SH'
#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${LLMUX_INSTALL_DIR:-$HOME/.local/share/llmux}"
BIN_DIR="${LLMUX_BIN_DIR:-$HOME/.local/bin}"

find_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    command -v "$PYTHON"
    return
  fi

  local candidate
  for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return
    fi
  done

  return 1
}

PYTHON_BIN="$(find_python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Python 3.10 or newer is required." >&2
  exit 1
fi

if ! "$PYTHON_BIN" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
then
  echo "Python 3.10 or newer is required. Found: $("$PYTHON_BIN" --version 2>&1)" >&2
  exit 1
fi

mkdir -p "$INSTALL_DIR" "$BIN_DIR"

if command -v rsync >/dev/null 2>&1; then
  rsync -a \
    --exclude '/.git' \
    --exclude '/.venv' \
    --exclude '/.runtime' \
    --exclude '/dist' \
    "$PACKAGE_DIR"/ "$INSTALL_DIR"/
else
  (
    cd "$PACKAGE_DIR"
    tar \
      --exclude './.git' \
      --exclude './.venv' \
      --exclude './.runtime' \
      --exclude './dist' \
      -cf - .
  ) | (
    cd "$INSTALL_DIR"
    tar -xf -
  )
fi

if [[ ! -f "$INSTALL_DIR/.env.common" && -f "$INSTALL_DIR/.env.common.example" ]]; then
  cp "$INSTALL_DIR/.env.common.example" "$INSTALL_DIR/.env.common"
fi

if [[ ! -f "$INSTALL_DIR/profiles.yaml" && -f "$INSTALL_DIR/profiles.example.yaml" ]]; then
  cp "$INSTALL_DIR/profiles.example.yaml" "$INSTALL_DIR/profiles.yaml"
fi

rm -rf "$INSTALL_DIR/.venv"
if ! "$PYTHON_BIN" -m venv "$INSTALL_DIR/.venv"; then
  if command -v uv >/dev/null 2>&1; then
    echo "python -m venv failed; retrying with uv venv --seed."
    rm -rf "$INSTALL_DIR/.venv"
    uv venv --seed --python "$PYTHON_BIN" "$INSTALL_DIR/.venv"
  else
    echo "Failed to create a Python virtual environment." >&2
    echo "On Debian/Ubuntu, install python3-venv and rerun this installer." >&2
    echo "Alternatively, install uv and rerun this installer." >&2
    exit 1
  fi
fi

if [[ ! -x "$INSTALL_DIR/.venv/bin/python" ]]; then
  echo "Failed to create a Python virtual environment." >&2
  exit 1
fi

VENV_PY="$INSTALL_DIR/.venv/bin/python"
"$VENV_PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install "$INSTALL_DIR"

WRAPPER="$BIN_DIR/llmux"
cat > "$WRAPPER" <<EOF_WRAPPER
#!/usr/bin/env sh
export LLMUX_ROOT="$INSTALL_DIR"
exec "$INSTALL_DIR/.venv/bin/llmux" "\$@"
EOF_WRAPPER
chmod +x "$WRAPPER"

echo "llmux installed to $INSTALL_DIR"
echo "Command installed at $WRAPPER"
echo "Edit $INSTALL_DIR/.env.common and $INSTALL_DIR/profiles.yaml before starting models."
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) echo "Add $BIN_DIR to PATH if the llmux command is not found." ;;
esac
INSTALL_SH
chmod +x "$PKG_DIR/install.sh"

if [[ "$PLATFORM" == linux-* ]]; then
  TARBALL="$RELEASE_DIR/${PKG_NAME}.tar.gz"
  RUNFILE="$RELEASE_DIR/${PKG_NAME}.run"

  tar -czf "$TARBALL" -C "$PKG_PARENT" "$PKG_NAME"

  cat > "$RUNFILE" <<'RUN_SH'
#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_LINE="$(awk '/^__LLMUX_ARCHIVE_BELOW__/ { print NR + 1; exit 0; }' "$0")"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

tail -n +"$ARCHIVE_LINE" "$0" | tar -xz -C "$TMP_DIR"
PKG_DIR="$(find "$TMP_DIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
exec "$PKG_DIR/install.sh" "$@"

__LLMUX_ARCHIVE_BELOW__
RUN_SH
  cat "$TARBALL" >> "$RUNFILE"
  chmod +x "$RUNFILE"
elif [[ "$PLATFORM" == macos-* ]]; then
  DMG_ROOT="$RELEASE_DIR/dmg-root"
  rm -rf "$DMG_ROOT"
  mkdir -p "$DMG_ROOT"
  cp -R "$PKG_DIR" "$DMG_ROOT/$PKG_NAME"
  cat > "$DMG_ROOT/Install llmux.command" <<'COMMAND_SH'
#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(find "$HERE" -maxdepth 1 -type d -name 'llmux-v*-macos-*' | head -n 1)"

if [[ -z "$PKG_DIR" ]]; then
  echo "Could not find the llmux installer payload." >&2
  exit 1
fi

exec "$PKG_DIR/install.sh"
COMMAND_SH
  chmod +x "$DMG_ROOT/Install llmux.command"
else
  echo "Unsupported platform: $PLATFORM" >&2
  exit 2
fi

echo "Built release assets for $PKG_NAME in $RELEASE_DIR"
