"""CLI runtime helpers — async glue, backend detection, output formatting."""

from __future__ import annotations

import asyncio
import json as _json
import sys
from typing import Any, AsyncIterator, Iterable

import typer

from tui.common import profile_store

BACKENDS = ("vllm", "llamacpp")


def detect_backend(name: str, *, override: str | None = None) -> str:
    """Resolve a profile name to its backend.

    `override` (from `--backend`) wins. Otherwise scan profiles.yaml and pick
    the unique backend that owns this name; ambiguity / not-found raises.
    """
    if override:
        if override not in BACKENDS:
            raise typer.BadParameter(
                f"unknown backend: {override} (choose from {', '.join(BACKENDS)})",
                param_hint="--backend",
            )
        if profile_store.load_profile(name, override) is None:
            raise typer.BadParameter(
                f"profile '{name}' not found in backend '{override}'",
                param_hint="PROFILE",
            )
        return override

    matches = [b for b in BACKENDS if profile_store.load_profile(name, b) is not None]
    if not matches:
        raise typer.BadParameter(
            f"profile '{name}' not found in profiles.yaml",
            param_hint="PROFILE",
        )
    if len(matches) > 1:
        raise typer.BadParameter(
            f"profile '{name}' exists in multiple backends ({', '.join(matches)}); "
            "disambiguate with --backend",
            param_hint="PROFILE",
        )
    return matches[0]


def run_async(coro):
    """Drive an async coroutine to completion, returning its result."""
    return asyncio.run(coro)


def stream_async(agen: AsyncIterator) -> int:
    """Drive an async generator that yields ('log', line) and exits with ('rc', n).

    Lines go to stdout. Returns the final rc (or 0 if none yielded).
    """

    async def _drive() -> int:
        rc = 0
        async for evt in agen:
            kind = evt[0]
            if kind == "log":
                print(evt[1], flush=True)
            elif kind == "rc":
                rc = int(evt[1])
        return rc

    try:
        return asyncio.run(_drive())
    except KeyboardInterrupt:
        return 130


def emit_json(data: Any) -> None:
    """Print a JSON payload (sorted keys, no trailing newline-magic)."""
    _json.dump(data, sys.stdout, ensure_ascii=False, indent=2, default=str)
    sys.stdout.write("\n")


def emit_table(rows: Iterable[dict], columns: list[str]) -> None:
    """Print a minimal padded table to stdout. No external deps."""
    rows = list(rows)
    if not rows:
        return
    widths = {c: len(c) for c in columns}
    for r in rows:
        for c in columns:
            widths[c] = max(widths[c], len(str(r.get(c, ""))))
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    sep = "  ".join("-" * widths[c] for c in columns)
    print(header)
    print(sep)
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns))
