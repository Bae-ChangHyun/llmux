#!/bin/bash
set -euo pipefail

if [[ -n "${EXTRA_PIP_PACKAGES:-}" ]]; then
    if [[ "$EXTRA_PIP_PACKAGES" =~ [[:alpha:]][[:alnum:]+.-]*://[^[:space:]]*@ ]] || \
       [[ "$EXTRA_PIP_PACKAGES" =~ [[:alpha:]][[:alnum:]+.-]*://[^[:space:]]*\? ]]; then
        echo "[entrypoint] ERROR: credential-bearing or query-string package URLs are not allowed" >&2
        exit 2
    fi
    requirements_file=$(mktemp)
    trap 'rm -f "$requirements_file"' EXIT
    IFS=' ' read -ra PACKAGES <<< "$EXTRA_PIP_PACKAGES"
    printf '%s\n' "${PACKAGES[@]}" > "$requirements_file"
    chmod 600 "$requirements_file"
    echo "[entrypoint] Installing configured extra packages"
    if pip install --disable-pip-version-check --no-cache-dir -r "$requirements_file"; then
        echo "[entrypoint] Extra packages installed successfully"
    else
        echo "[entrypoint] ERROR: Failed to install extra packages"
        exit 1
    fi
fi

exec vllm serve "$@"
