from __future__ import annotations

import os
import re
import ssl
import urllib.parse
import urllib.request

_CA_CANDIDATES = (
    "/etc/pki/tls/certs/ca-bundle.crt",     # RHEL / CentOS / Fedora
    "/etc/ssl/certs/ca-certificates.crt",   # Debian / Ubuntu / Alpine
    "/etc/ssl/cert.pem",                    # macOS / OpenSSL default
)

_cached: ssl.SSLContext | None = None
_SENSITIVE_HEADERS = {"authorization", "cookie", "proxy-authorization"}


def same_origin(left: str, right: str) -> bool:
    def origin(url: str) -> tuple[str, str, int | None] | None:
        try:
            parsed = urllib.parse.urlsplit(url)
            scheme = parsed.scheme.lower()
            hostname = (parsed.hostname or "").lower()
            port = parsed.port
        except ValueError:
            return None
        if scheme not in {"http", "https"} or not hostname:
            return None
        if port is None:
            port = {"http": 80, "https": 443}.get(scheme)
        return scheme, hostname, port

    source = origin(left)
    return source is not None and source == origin(right)


def redact_sensitive_text(text: str, secrets: tuple[str, ...] = ()) -> str:
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "<redacted>")
    redacted = re.sub(
        r"(?i)(\b(?:authorization|proxy-authorization)\s*:\s*bearer\s+)[^\s,;]+",
        r"\1<redacted>",
        redacted,
    )
    return re.sub(
        r"(?i)(\b(?:hf_token|access_token|token)\s*[=:]\s*)[^\s,;]+",
        r"\1<redacted>",
        redacted,
    )


class _SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not same_origin(req.full_url, newurl):
            return None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def get_ssl_context() -> ssl.SSLContext:
    global _cached
    if _cached is not None:
        return _cached

    context = ssl.create_default_context()
    candidates = [path for path in _CA_CANDIDATES if os.path.exists(path)]
    try:
        import certifi  # type: ignore[import-not-found]

        certifi_path = certifi.where()
        if not certifi_path or not os.path.exists(certifi_path):
            raise RuntimeError(f"certifi CA bundle does not exist: {certifi_path}")
        candidates.append(certifi_path)
    except ImportError:
        certifi_path = ""

    failures: list[str] = []
    for cafile in dict.fromkeys(candidates):
        try:
            context.load_verify_locations(cafile=cafile)
        except (OSError, ssl.SSLError) as exc:
            failures.append(f"{cafile}: {exc}")
    if failures:
        raise RuntimeError("Could not load CA bundle(s): " + "; ".join(failures))
    if not context.get_ca_certs():
        searched = ", ".join(_CA_CANDIDATES)
        if certifi_path:
            searched += f", {certifi_path}"
        raise RuntimeError(f"No CA certificates loaded; searched {searched}")

    _cached = context
    return _cached


def open_url(request: urllib.request.Request | str, *, timeout: float):
    context = get_ssl_context()
    if isinstance(request, str):
        return urllib.request.urlopen(request, timeout=timeout, context=context)
    opener = urllib.request.build_opener(
        _SameOriginRedirectHandler(),
        urllib.request.HTTPSHandler(context=context),
    )
    return opener.open(request, timeout=timeout)
