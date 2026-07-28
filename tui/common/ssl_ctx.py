"""Shared HTTPS context with a working CA bundle.

Some Python builds (uv-managed CPython on RHEL/CentOS) default to a
`/etc/ssl/cert.pem` that does not exist, so every HTTPS call fails with
CERTIFICATE_VERIFY_FAILED until an existing bundle is passed explicitly.
"""

from __future__ import annotations

import os
import ssl

_CA_CANDIDATES = (
    "/etc/pki/tls/certs/ca-bundle.crt",     # RHEL / CentOS / Fedora
    "/etc/ssl/certs/ca-certificates.crt",   # Debian / Ubuntu / Alpine
    "/etc/ssl/cert.pem",                    # macOS / OpenSSL default
)

_cached: ssl.SSLContext | None = None


def get_ssl_context() -> ssl.SSLContext:
    global _cached
    if _cached is not None:
        return _cached

    candidates: list[str] = []
    try:
        import certifi  # type: ignore[import-not-found]

        candidates.append(certifi.where())
    except Exception:  # noqa: BLE001 — certifi is optional
        pass
    candidates.extend(_CA_CANDIDATES)

    for cafile in candidates:
        if not cafile or not os.path.exists(cafile):
            continue
        try:
            _cached = ssl.create_default_context(cafile=cafile)
            return _cached
        except Exception:  # noqa: BLE001 — try the next candidate
            continue

    _cached = ssl.create_default_context()
    return _cached
