"""Subprocess execution helpers for backend operations."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

from .backend_common import SCRIPT_DIR


async def run_command(*args: str, timeout: float = 30) -> tuple[int, str]:
    """Run a command and return (returncode, combined stdout+stderr)."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(SCRIPT_DIR),
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "Command timed out"
    return proc.returncode or 0, (stdout or b"").decode(errors="replace")


async def run_command_with_options(
    *args: str,
    timeout: float = 30,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Run a command with explicit cwd/env and return combined output."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(cwd or SCRIPT_DIR),
        env=env,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "Command timed out"
    return proc.returncode or 0, (stdout or b"").decode(errors="replace")


async def stream_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
):
    """Yield stdout/stderr lines followed by the final return code."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(cwd or SCRIPT_DIR),
        env=env,
    )
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            yield ("log", line.decode(errors="replace").rstrip("\n"))
        await proc.wait()
        yield ("rc", proc.returncode or 0)
    except asyncio.CancelledError:
        # Async generator cleanup (interpreter shutdown, athrow during
        # asyncio.run finalization, or explicit caller cancel) lands here.
        # If the underlying compose/docker process has already exited,
        # proc.kill() raises ProcessLookupError that escapes as an
        # "unhandled exception during asyncio.run() shutdown" traceback.
        # Suppress these specifically — actual signal-delivery failures
        # for live processes still surface via wait()'s return code.
        with contextlib.suppress(ProcessLookupError, OSError):
            proc.kill()
        with contextlib.suppress(ProcessLookupError, OSError):
            await proc.wait()
        raise
