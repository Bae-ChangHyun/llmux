"""Plain-text terminal dashboard — the unified dashboard without the Textual TUI.

Two entry points share this renderer: pressing `t` in the TUI (which suspends
Textual and runs this in the normal terminal) and `llmux top` (which launches it
directly, no TUI at all). Auto-refreshes once a second; exits on `q` or Ctrl+C.
"""

from __future__ import annotations

import asyncio
import sys
from time import monotonic

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from tui.backends.llamacpp.adapter import LlamacppAdapter
from tui.backends.vllm.adapter import VllmAdapter
from tui.common import docker as common_docker
from tui.common.i18n import t
from tui.common.metrics import ThroughputTracker, fetch_token_counters

_VLLM_BLUE = "#4c8dff"


async def _collect(tracker: ThroughputTracker, now: float):
    try:
        running = await common_docker.running_container_names()
    except Exception:
        running = set()
    rows = []
    for adapter in (VllmAdapter(), LlamacppAdapter()):
        try:
            rows.extend(adapter.rows(running))
        except Exception:
            pass
    rows.sort(key=lambda r: (not r.running, r.backend, r.profile_name))

    tps: dict[str, float] = {}
    for r in rows:
        if not (r.running and r.port):
            continue
        key = f"{r.backend}:{r.profile_name}"
        counters = await fetch_token_counters(r.port)
        if counters is None:
            tracker.forget(key)
            continue
        rate = tracker.update(key, counters, now)
        if rate is not None:
            tps[key] = rate[1]
    gpus = await common_docker.get_gpu_info()
    return rows, tps, gpus


def render(rows, tps, gpus) -> Group:
    v_run = sum(1 for r in rows if r.backend == "vllm" and r.running)
    v_total = sum(1 for r in rows if r.backend == "vllm")
    l_run = sum(1 for r in rows if r.backend == "llamacpp" and r.running)
    l_total = sum(1 for r in rows if r.backend == "llamacpp")
    header = Text.from_markup(
        f"[b]llmux[/b]  [{_VLLM_BLUE}]vLLM[/] {v_run}/{v_total}   "
        f"[green]llama.cpp[/] {l_run}/{l_total}"
    )

    table = Table(expand=True, pad_edge=False)
    for col in ("Status", "Backend", "Profile", "Port", "tok/s", "Model"):
        table.add_column(col, no_wrap=True)
    for r in rows:
        key = f"{r.backend}:{r.profile_name}"
        status = Text("● running", "green") if r.running else Text("○ stopped", "dim")
        backend = (
            Text("vLLM", _VLLM_BLUE) if r.backend == "vllm" else Text("llama.cpp", "green")
        )
        rate = tps.get(key)
        tok = f"{rate:.1f}" if rate is not None else "—"
        model = r.model.split("/")[-1] if "/" in r.model else (r.model or "—")
        table.add_row(status, backend, r.profile_name, str(r.port or "—"), tok, model)

    gpu = Text.from_markup(common_docker.format_gpu_bar(gpus))
    hint = Text.from_markup(t("[dim]q: back · auto-refresh 1s[/]", "[dim]q: 뒤로 · 1초마다 갱신[/]"))
    return Group(header, Text(), table, Text(), gpu, Text(), hint)


async def run_plain_dashboard(interval: float = 1.0) -> None:
    """Blocking-ish async loop that renders the dashboard as plain text.

    Reads a single keypress per tick (cbreak) so `q` returns immediately;
    Ctrl+C also exits. Terminal settings are always restored.
    """
    console = Console()
    tracker = ThroughputTracker()

    old_attrs = None
    fd = None
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_attrs = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except Exception:
        old_attrs = None  # not a real tty (piped/CI) — no key reads, Ctrl+C only

    try:
        with Live(console=console, screen=True, auto_refresh=False) as live:
            while True:
                rows, tps, gpus = await _collect(tracker, monotonic())
                live.update(render(rows, tps, gpus))
                live.refresh()
                waited = 0.0
                while waited < interval:
                    await asyncio.sleep(0.1)
                    waited += 0.1
                    if fd is not None and _key_ready():
                        ch = sys.stdin.read(1)
                        if ch in ("q", "Q", "\x03", "\x1b"):
                            return
    except (KeyboardInterrupt, asyncio.CancelledError):
        return
    finally:
        if old_attrs is not None and fd is not None:
            try:
                import termios

                termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
            except Exception:
                pass


def _key_ready() -> bool:
    import select

    return bool(select.select([sys.stdin], [], [], 0)[0])


def run_cli() -> None:
    """`llmux top` entry point."""
    try:
        asyncio.run(run_plain_dashboard())
    except KeyboardInterrupt:
        pass
