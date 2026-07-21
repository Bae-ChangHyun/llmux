"""Serialize/parse for llmux's "disabled config parameter" comment markers.

A disabled parameter is kept in the config file as a trailing *comment* line:

    # llmux:disabled <key>: <yaml-inline-value>

vLLM's `vllm serve` reads the config YAML directly and llama-server's flags are
rendered from it, so a disabled param cannot live as a normal key — it would be
passed to the server. A comment is inert to every YAML/flag parser, so the
param is preserved (for re-enabling) without ever reaching the server.
"""

from __future__ import annotations

import re
from typing import Any

import yaml

_MARKER_PREFIX = "# llmux:disabled "
# Key is a config flag name (no colons); everything after the first ": " is the
# inline-YAML value.
_MARKER_RE = re.compile(r"^#\s*llmux:disabled\s+([^:]+):\s?(.*)$")


def _inline(value: Any) -> str:
    # A marker is exactly one line, so the serialization must never wrap: the
    # default width=80 used to line-break long strings / big lists, and taking
    # only the first physical line silently truncated the value (and corrupted
    # list/dict types on re-enable). width=10**9 disables wrapping.
    dumped = yaml.safe_dump(
        value,
        default_flow_style=True,
        allow_unicode=True,
        sort_keys=False,
        width=10**9,
    )
    # Drop the bare `...` document-end marker scalars emit on their own line,
    # then defensively space-join — a marker line must stay single-line (the
    # only case that still has a real newline here is a string value that
    # itself contains one, which config flags never do).
    lines = [ln for ln in dumped.strip().split("\n") if ln.strip() != "..."]
    return " ".join(lines)


def render_disabled_markers(disabled: dict[str, Any]) -> str:
    """Trailing comment block for a config file (empty string when none)."""
    lines = [f"{_MARKER_PREFIX}{key}: {_inline(value)}" for key, value in disabled.items()]
    return ("\n".join(lines) + "\n") if lines else ""


def dump_active_config(existing_text: str | None, data: dict[str, Any]) -> str:
    """Serialize `data` as the active-parameter YAML, keeping the user's own
    comments (header, inline, trailing blocks) for any key that survives.

    PyYAML can't round-trip comments, so editing a config used to erase every
    hand-written `#` note. When the prior file carried comments we merge `data`
    into it via ruamel's round-trip loader instead: values are replaced in
    place (their attached comment stays), removed keys drop out, new keys append
    at the end. A file with no comments — the common case — takes the original
    plain PyYAML dump so its output stays byte-identical.

    `existing_text` is the raw prior file (disabled markers and all); callers
    pass the trailing disabled block separately via `render_disabled_markers`.
    """
    active = strip_disabled_markers(existing_text or "")
    plain = lambda: yaml.dump(  # noqa: E731 — one-liner fallback, used 3×
        data, default_flow_style=False, allow_unicode=True, sort_keys=False
    )
    if "#" not in active:
        return plain()

    try:
        from io import StringIO

        from ruamel.yaml import YAML
    except ImportError as exc:
        raise RuntimeError(
            "ruamel.yaml is required to preserve config comments but is not "
            "installed; a plain dump would silently drop disabled-param markers. "
            "Reinstall dependencies (uv sync)."
        ) from exc

    ry = YAML()
    ry.preserve_quotes = True
    ry.width = 10**9
    try:
        existing = ry.load(active)
    except Exception:  # noqa: BLE001 — malformed prior file: don't block the save
        return plain()
    if not isinstance(existing, dict):
        return plain()

    for key in [k for k in existing if k not in data]:
        del existing[key]
    for key, value in data.items():
        existing[key] = value

    buf = StringIO()
    ry.dump(existing, buf)
    return buf.getvalue()


def parse_disabled_markers(text: str) -> dict[str, Any]:
    """Extract disabled params from a raw config file's marker lines."""
    out: dict[str, Any] = {}
    for line in text.splitlines():
        m = _MARKER_RE.match(line.strip())
        if not m:
            continue
        key = m.group(1).strip()
        raw = m.group(2)
        try:
            out[key] = yaml.safe_load(raw)
        except yaml.YAMLError:
            out[key] = raw
    return out


def strip_disabled_markers(text: str) -> str:
    """Return `text` with any disabled-marker lines removed."""
    kept = [
        line for line in text.splitlines()
        if not _MARKER_RE.match(line.strip())
    ]
    result = "\n".join(kept)
    return result + "\n" if result and not result.endswith("\n") else result
