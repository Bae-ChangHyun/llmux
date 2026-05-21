#!/bin/sh
# llmux installer
#
#   curl -fsSL https://raw.githubusercontent.com/Bae-ChangHyun/llmux/main/install.sh | sh
#
# Clones llmux, installs its dependencies, and puts the `llmux` command on PATH.
# The checkout stays a live git repo, so `git pull` applies updates without a
# reinstall. Override the install location with the LLMUX_DIR environment var.
set -eu

REPO_URL="https://github.com/Bae-ChangHyun/llmux.git"
INSTALL_DIR="${LLMUX_DIR:-$HOME/.llmux}"

# ── output helpers (colour only on a real terminal) ──────────────────────────
if [ -t 1 ]; then
    C_INFO='\033[1;36m'; C_OK='\033[1;32m'; C_WARN='\033[1;33m'
    C_ERR='\033[1;31m'; C_BOLD='\033[1m'; C_OFF='\033[0m'
else
    C_INFO=''; C_OK=''; C_WARN=''; C_ERR=''; C_BOLD=''; C_OFF=''
fi
info() { printf '%b▸%b %s\n' "$C_INFO" "$C_OFF" "$1"; }
ok()   { printf '%b✓%b %s\n' "$C_OK" "$C_OFF" "$1"; }
warn() { printf '%b⚠%b %s\n' "$C_WARN" "$C_OFF" "$1"; }
err()  { printf '%b✗%b %s\n' "$C_ERR" "$C_OFF" "$1" >&2; }

# ── 1. prerequisites ─────────────────────────────────────────────────────────
if ! command -v git >/dev/null 2>&1; then
    err "git is required but not found. Install git, then re-run this installer."
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    info "uv not found — installing it from astral.sh ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv's installer drops the binary in one of these; surface it for this run.
    for d in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
        if [ -x "$d/uv" ]; then PATH="$d:$PATH"; fi
    done
    export PATH
fi
if ! command -v uv >/dev/null 2>&1; then
    err "uv installation failed — install it manually: https://docs.astral.sh/uv/"
    exit 1
fi

# ── 2. clone (or fast-forward an existing checkout) ──────────────────────────
if [ -d "$INSTALL_DIR/.git" ]; then
    info "Existing llmux checkout at $INSTALL_DIR — updating ..."
    git -C "$INSTALL_DIR" pull --ff-only
elif [ -e "$INSTALL_DIR" ]; then
    err "$INSTALL_DIR exists but is not a llmux checkout."
    err "Remove it, or set LLMUX_DIR to another path, then re-run."
    exit 1
else
    info "Cloning llmux into $INSTALL_DIR ..."
    git clone "$REPO_URL" "$INSTALL_DIR"
fi

# ── 3. install dependencies + the `llmux` command ────────────────────────────
# --editable keeps the command pointed at the checkout, so a later `git pull`
# takes effect with no reinstall. --force makes re-running this script a no-op.
info "Installing dependencies and the llmux command ..."
( cd "$INSTALL_DIR" && uv tool install --editable . --force )
uv tool update-shell >/dev/null 2>&1 || true

# ── 4. next steps ────────────────────────────────────────────────────────────
echo
ok "llmux installed at $INSTALL_DIR"
echo
printf '  Next:\n'
printf '    1. Shared settings — HF token, cache + model dirs:\n'
printf '       %bcd %s%b  then  %bcp .env.common.example .env.common%b  and edit it\n' \
    "$C_BOLD" "$INSTALL_DIR" "$C_OFF" "$C_BOLD" "$C_OFF"
printf '    2. Profiles:  %bcp profiles.example.yaml profiles.yaml%b\n' "$C_BOLD" "$C_OFF"
printf '    3. Launch:    %bllmux%b\n' "$C_BOLD" "$C_OFF"
echo
if ! command -v llmux >/dev/null 2>&1; then
    warn "The 'llmux' command is not on PATH in this shell yet."
    warn "Open a new terminal, or run:  export PATH=\"\$HOME/.local/bin:\$PATH\""
fi
