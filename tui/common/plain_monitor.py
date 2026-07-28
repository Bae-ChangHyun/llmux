"""Plain-terminal btop-style monitor — GPUs always, plus every running model.

Reached via `llmux top [profile]` (no TUI) or `t` on a dashboard row (which
suspends Textual and runs this). Shares its rendering with the Textual `v`
monitor through `monitor_render.render_dashboard`, so both show the same view.

It is a *system* monitor: the GPU panel is drawn whether or not anything is
running, and the running-model list is re-scanned every tick so a model started
elsewhere shows up on its own. Keys: `q`/Esc quit, `p` pause, `r` reset peaks,
`+`/`-` poll interval, `l` language.
"""

from __future__ import annotations

import asyncio
import os
import sys
from time import monotonic

from rich.console import Console

from tui.common import docker as common_docker
from tui.common.adapter import DashboardRow
from tui.common.i18n import t
from tui.common.metrics import fetch_snapshot
from tui.common.monitor_render import ModelEntry, MonitorState, render_dashboard

_MIN_INTERVAL = 0.25
_MAX_INTERVAL = 5.0


def _key_ready() -> bool:
    import select

    return bool(select.select([sys.stdin], [], [], 0)[0])


def _toggle_lang() -> None:
    os.environ["LLMUX_LANG"] = "en" if os.environ.get("LLMUX_LANG") == "ko" else "ko"


async def _running_rows() -> tuple[list[DashboardRow], list[str]]:
    from tui.backends.llamacpp.adapter import LlamacppAdapter
    from tui.backends.vllm.adapter import VllmAdapter

    from tui.common.i18n import t

    errors: list[str] = []
    try:
        running = await common_docker.running_container_names()
    except Exception as exc:
        return [], [t(f"Docker status scan failed: {exc}",
                      f"Docker 상태 스캔 실패: {exc}")]
    rows: list[DashboardRow] = []
    for adapter in (VllmAdapter(), LlamacppAdapter()):
        label = type(adapter).__name__.replace("Adapter", "")
        try:
            rows.extend(adapter.rows(running))
        except Exception as exc:
            errors.append(t(f"{label} scan failed: {exc}",
                            f"{label} 스캔 실패: {exc}"))
    return [r for r in rows if r.running], errors


async def sample_entries(
    focus: str | None, states: dict[str, MonitorState], now: float, lag_ms: float
) -> tuple[list[ModelEntry], list[str]]:
    rows, errors = await _running_rows()
    if focus:
        rows = [r for r in rows if r.profile_name == focus]
    entries: list[ModelEntry] = []
    for row in rows:
        snap = await fetch_snapshot(row.port) if row.port else None
        state = states.setdefault(row.profile_name, MonitorState())
        entries.append(ModelEntry(row, snap, state, state.update(snap, now, lag_ms)))
    return entries, errors


async def run_plain_monitor(focus: str | None = None, interval: float = 1.0) -> None:
    from rich.live import Live

    console = Console()
    states: dict[str, MonitorState] = {}
    started = monotonic()
    paused = False
    fd = None
    old_attrs = None
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_attrs = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except Exception:
        fd = None
        old_attrs = None

    last: dict = {"entries": [], "gpus": [], "pcie": {}, "lag": 0.0, "ready": False,
                  "notices": []}
    try:
        with Live(console=console, screen=True, auto_refresh=False) as live:
            while True:
                if not paused:
                    t0 = monotonic()
                    gpus = await common_docker.get_gpu_info()
                    pcie = await common_docker.get_pcie_stats()
                    lag_ms = (monotonic() - t0) * 1000
                    entries, notices = await sample_entries(
                        focus, states, monotonic(), lag_ms
                    )
                    last.update(entries=entries, gpus=gpus, pcie=pcie, lag=lag_ms,
                                ready=True, notices=notices)
                if last["ready"]:
                    live.update(render_dashboard(
                        last["entries"], last["gpus"], last["pcie"], console.size.width,
                        paused=paused, interval=interval,
                        uptime=monotonic() - started, lag_ms=last["lag"],
                        notices=last["notices"],
                    ))
                    live.refresh()

                waited = 0.0
                while waited < interval:
                    await asyncio.sleep(0.1)
                    waited += 0.1
                    if fd is not None and _key_ready():
                        ch = sys.stdin.read(1)
                        if ch in ("q", "Q", "\x03", "\x1b"):
                            return
                        if ch in ("p", "P"):
                            paused = not paused
                        elif ch in ("r", "R"):
                            for st in states.values():
                                st.reset_peaks()
                        elif ch in ("+", "="):
                            interval = max(_MIN_INTERVAL, round(interval - 0.25, 2))
                        elif ch in ("-", "_"):
                            interval = min(_MAX_INTERVAL, round(interval + 0.25, 2))
                        elif ch in ("l", "L"):
                            _toggle_lang()
                        break
    except (KeyboardInterrupt, asyncio.CancelledError):
        return
    finally:
        if old_attrs is not None and fd is not None:
            try:
                import termios

                termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
            except Exception:
                pass


async def _resolve_and_run(profile: str | None) -> int:
    if profile:
        rows, errors = await _running_rows()
        if errors:
            for line in errors:
                print(line, file=sys.stderr)
            print(t(f"Cannot tell whether '{profile}' is running.",
                    f"'{profile}' 의 실행 여부를 확인할 수 없습니다."), file=sys.stderr)
            return 2
        if not any(r.profile_name == profile for r in rows):
            names = ", ".join(r.profile_name for r in rows) or "—"
            print(t(f"'{profile}' is not running. Running: {names}",
                    f"'{profile}' 은 실행 중이 아닙니다. 실행 중: {names}"))
            return 1
    await run_plain_monitor(profile)
    return 0


def run_cli(profile: str | None = None) -> int:
    """`llmux top [profile]` entry point."""
    try:
        return asyncio.run(_resolve_and_run(profile))
    except KeyboardInterrupt:
        return 0
