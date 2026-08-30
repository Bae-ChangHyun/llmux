"""CLI runtime helpers — async glue, backend detection, output formatting."""

from __future__ import annotations

import asyncio
import contextlib
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

    Explicitly closes the generator before asyncio.run() shutdown so that
    the inner subprocess wrappers can run their cleanup deterministically.
    Backend code may call `proc.kill()` on already-exited processes during
    cancellation, raising ProcessLookupError; we swallow those (and the
    cooperating CancelledError) so they don't leak as unraisable
    exceptions to stderr.
    """

    async def _drive() -> int:
        rc = 0
        try:
            async for evt in agen:
                kind = evt[0]
                if kind == "log":
                    print(evt[1], flush=True)
                elif kind == "rc":
                    rc = int(evt[1])
        finally:
            with contextlib.suppress(
                ProcessLookupError, OSError, asyncio.CancelledError
            ):
                await agen.aclose()
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




async def gather_conflict_warnings(profile_name: str, backend: str) -> list[str]:
    """Pre-flight: run the same cross-backend port/GPU checks the TUI dashboard
    runs before starting a container.

    Mirrors `DashboardScreen._check_and_confirm`: aggregates llmux-managed rows
    from both backend adapters, then applies `port_conflicts`, `gpu_conflicts`,
    and `external_port_conflicts` against a fresh `docker ps` snapshot. Returns
    a list of human-readable warning lines (empty = no conflicts). Each line is
    pre-categorized so the caller can just print it.
    """
    # Imports live inside the function so the CLI can `--help` without paying
    # the Textual / docker import cost.
    from tui.backends.llamacpp.adapter import LlamacppAdapter
    from tui.backends.vllm.adapter import VllmAdapter
    from tui.common import docker as common_docker
    from tui.common.conflicts import (
        external_port_conflicts,
        gpu_conflicts,
        port_conflicts,
    )

    probe_warnings: list[str] = []
    try:
        running = await common_docker.running_container_names()
    except Exception as exc:
        running = set()
        probe_warnings.append(
            "could not enumerate running containers via `docker ps` "
            f"({exc}) — pre-flight conflict scan may be incomplete"
        )

    rows = []
    try:
        rows.extend(VllmAdapter().rows(running))
    except Exception as exc:
        probe_warnings.append(f"vLLM profile scan failed: {exc}")
    try:
        rows.extend(LlamacppAdapter().rows(running))
    except Exception as exc:
        probe_warnings.append(f"llama.cpp profile scan failed: {exc}")

    target = next(
        (r for r in rows if r.backend == backend and r.profile_name == profile_name),
        None,
    )
    if target is None:
        # Profile vanished between detect_backend and now; let the backend's
        # own start path surface the error.
        return [f"port probe warning: {m}" for m in probe_warnings]

    warnings: list[str] = [f"port probe warning: {m}" for m in probe_warnings]
    for m in port_conflicts(target, rows):
        warnings.append(f"port conflict (llmux): {m}")
    for m in gpu_conflicts(target, rows):
        warnings.append(f"GPU conflict: {m}")
    try:
        ext_ports = await common_docker.running_container_ports()
    except Exception as exc:
        ext_ports = {}
        warnings.append(
            "port probe warning: could not inspect running container ports "
            f"({exc})"
        )
    for m in external_port_conflicts(target, rows, ext_ports):
        warnings.append(f"port conflict (external): {m}")
    return warnings


def partition_conflict_warnings(warnings: list[str]) -> tuple[list[str], list[str]]:
    soft: list[str] = []
    hard: list[str] = []
    for warning in warnings:
        if warning.startswith("GPU conflict:"):
            soft.append(warning)
        else:
            hard.append(warning)
    return hard, soft


async def docker_logs_once(container_name: str, *, tail: int) -> int:
    """Print the last `tail` log lines for a container and exit (no follow).

    Used by `llmux logs --no-follow` so the non-follow path goes through the
    same async-subprocess wrapper the follow path uses, instead of a bare
    `subprocess.run` call.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "logs", "--tail", str(tail), container_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError:
        print("Error: docker executable not found", file=sys.stderr)
        return 127
    if proc.stdout is not None:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            print(line.decode("utf-8", errors="replace").rstrip("\n"), flush=True)
    await proc.wait()
    return proc.returncode or 0


async def docker_logs_follow(container_name: str, *, tail: int) -> int:
    """Stream `docker logs -f` and return its exit code.

    The CLI needs the child's status: with `stderr=STDOUT`, a missing container
    prints "Error: No such container" as a log line and the stream simply ends,
    so a hardcoded 0 would report success for a container that never existed.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "logs", "-f", "--tail", str(tail), container_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError:
        print("Error: docker executable not found", file=sys.stderr)
        return 127
    try:
        if proc.stdout is not None:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                print(line.decode("utf-8", errors="replace").rstrip("\n"), flush=True)
        await proc.wait()
        return proc.returncode or 0
    finally:
        if proc.returncode is None:
            try:
                proc.kill()
            except (ProcessLookupError, OSError):
                pass
            try:
                await proc.wait()
            except (asyncio.CancelledError, ProcessLookupError, OSError):
                pass
