#!/bin/bash
# Wrapper script for vllm container entrypoint
# Installs additional pip packages before starting vllm serve

if [[ -n "$EXTRA_PIP_PACKAGES" ]]; then
    echo "[entrypoint] Installing extra packages: $EXTRA_PIP_PACKAGES"
    IFS=' ' read -ra PACKAGES <<< "$EXTRA_PIP_PACKAGES"
    if pip install --disable-pip-version-check --no-cache-dir -- "${PACKAGES[@]}"; then
        echo "[entrypoint] Extra packages installed successfully"
    else
        echo "[entrypoint] ERROR: Failed to install extra packages"
        exit 1
    fi
fi

exec vllm serve "$@"
