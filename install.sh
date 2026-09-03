#!/bin/sh
# llmux installer
#
#   curl -fsSL https://raw.githubusercontent.com/Changroro/llmux/main/install.sh | sh
#
# Clones llmux, installs its dependencies, and puts the `llmux` command on PATH.
# The checkout stays a live git repo, so `git pull` applies updates without a
# reinstall. Override the install location with the LLMUX_DIR environment var.
# Re-running this script when llmux is already installed updates the existing
# checkout in place (regardless of LLMUX_DIR) — set LLMUX_FORCE_RELOCATE=1 to
# override and relocate the install to LLMUX_DIR instead.
set -eu

REPO_URL="https://github.com/Changroro/llmux.git"
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

# ── 2. detect an existing install — update only, never relocate ──────────────
EXISTING_DIR=""
if ! tool_dir=$(uv tool dir); then
    err "'uv tool dir' failed — cannot tell whether llmux is already installed."
    err "Installing here would repoint an existing editable install at $INSTALL_DIR."
    exit 1
fi
receipt="$tool_dir/llmux/uv-receipt.toml"
if [ -f "$receipt" ]; then
    EXISTING_DIR=$(sed -n 's/^[[:space:]]*requirements[[:space:]]*=.*editable[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' \
        "$receipt" | head -n1)
    if [ -z "$EXISTING_DIR" ]; then
        err "llmux appears installed ($receipt) but the editable path could not be"
        err "parsed — uv's receipt format may have changed. Continuing would"
        err "repoint the command at $INSTALL_DIR. Run 'uv tool uninstall llmux'"
        err "first, or set LLMUX_FORCE_RELOCATE=1 to install here anyway."
        [ "${LLMUX_FORCE_RELOCATE:-}" = "1" ] || exit 1
    fi
fi

if [ -n "$EXISTING_DIR" ] && [ "${LLMUX_FORCE_RELOCATE:-}" != "1" ]; then
    if [ ! -d "$EXISTING_DIR/.git" ]; then
        err "llmux is already installed (editable) at $EXISTING_DIR, but that"
        err "path is no longer a git repository — cannot update in place."
        err "Restore the checkout, or run 'uv tool uninstall llmux' and re-run."
        exit 1
    fi
    info "llmux is already installed (editable) at $EXISTING_DIR — updating in place."
    if [ "$EXISTING_DIR" != "$INSTALL_DIR" ]; then
        info "LLMUX_DIR=$INSTALL_DIR is ignored once installed; set LLMUX_FORCE_RELOCATE=1 to override."
    fi
    if ! git -C "$EXISTING_DIR" pull --ff-only; then
        err "Could not fast-forward $EXISTING_DIR — the checkout has diverged."
        err "Resolve it manually (check 'git -C $EXISTING_DIR status'), then re-run."
        exit 1
    fi
    ( cd "$EXISTING_DIR" && uv tool install --editable . --force )
    uv tool update-shell >/dev/null 2>&1 || warn "uv tool update-shell failed — PATH may need a manual entry."
    echo
    ok "llmux updated at $EXISTING_DIR"
    exit 0
fi

if [ -n "$EXISTING_DIR" ] && [ "${LLMUX_FORCE_RELOCATE:-}" = "1" ]; then
    warn "LLMUX_FORCE_RELOCATE=1 — relocating from $EXISTING_DIR to $INSTALL_DIR."
fi

# ── 3. clone (or fast-forward an existing checkout) ──────────────────────────
if [ -d "$INSTALL_DIR/.git" ]; then
    if ! grep -q '^name = "llmux"' "$INSTALL_DIR/pyproject.toml" 2>/dev/null; then
        err "$INSTALL_DIR is a git repository but not llmux."
        err "Remove it, or set LLMUX_DIR to another path, then re-run."
        exit 1
    fi
    info "Existing llmux checkout at $INSTALL_DIR — updating ..."
    if ! git -C "$INSTALL_DIR" pull --ff-only; then
        err "Could not fast-forward $INSTALL_DIR — the checkout has diverged."
        err "Resolve it manually (check 'git -C $INSTALL_DIR status'), then re-run."
        exit 1
    fi
elif [ -e "$INSTALL_DIR" ]; then
    err "$INSTALL_DIR exists but is not a llmux checkout."
    err "Remove it, or set LLMUX_DIR to another path, then re-run."
    exit 1
else
    info "Cloning llmux into $INSTALL_DIR ..."
    git clone "$REPO_URL" "$INSTALL_DIR"
fi

# ── 4. install dependencies + the `llmux` command ────────────────────────────
# --editable keeps the command pointed at the checkout, so a later `git pull`
# takes effect with no reinstall. --force makes re-running this script a no-op.
info "Installing dependencies and the llmux command ..."
( cd "$INSTALL_DIR" && uv tool install --editable . --force )
uv tool update-shell >/dev/null 2>&1 || warn "uv tool update-shell failed — PATH may need a manual entry."

# ── 5. next steps ────────────────────────────────────────────────────────────
echo
ok "llmux installed at $INSTALL_DIR"
echo
printf '  Next:\n'
printf '    1. Run %bllmux%b — on first launch it walks you through shared\n' "$C_BOLD" "$C_OFF"
printf '       settings (HF token, cache + model dirs) and writes .env.common.\n'
printf '    2. Create your first profile from the dashboard (press %bn%b), or\n' "$C_BOLD" "$C_OFF"
printf '       headless: %bllmux profile quick-setup <hf-model-id>%b\n' "$C_BOLD" "$C_OFF"
echo
if ! command -v llmux >/dev/null 2>&1; then
    warn "The 'llmux' command is not on PATH in this shell yet."
    warn "Open a new terminal, or run:  export PATH=\"\$HOME/.local/bin:\$PATH\""
fi
