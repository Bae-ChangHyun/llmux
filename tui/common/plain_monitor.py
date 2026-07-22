"""Plain-text btop-style monitor for one running model — the `v`-key monitor
without the Textual TUI.

Reached via `llmux top [profile]` (no TUI) or `t` on a running row in the
dashboard (which suspends Textual and runs this). Auto-refreshes once a second;
`q` or Ctrl+C exits. Renders throughput sparklines, request queue, KV-cache
usage, latency, and per-GPU bars — the same metrics as the TUI monitor.
"""

from __future__ import annotations

import asyncio
import sys
from collections import deque
from time import monotonic

from rich.console import Console, Group
from rich.text import Text

from tui.common import docker as common_docker
from tui.common.adapter import DashboardRow
from tui.common.i18n import t
from tui.common.metrics import ServerMetrics, fetch_server_metrics

_SPARK = "▁▂▃▄▅▆▇█"
_HISTORY = 48
_VLLM_BLUE = "#4c8dff"


def _sparkline(values, width: int = 44) -> str:
    vals = list(values)[-width:]
    if not vals:
        return ""
    hi = max(vals) or 1.0
    return "".join(_SPARK[min(len(_SPARK) - 1, int(v / hi * (len(_SPARK) - 1)))] for v in vals)


def _bar(ratio: float | None, width: int = 24, color: str = "cyan") -> str:
    if ratio is None:
        return "[dim]" + "░" * width + "[/]"
    ratio = max(0.0, min(1.0, ratio))
    filled = int(round(ratio * width))
    return f"[{color}]{'█' * filled}[/][dim]{'░' * (width - filled)}[/]"


class MonitorState:
    """Rolling history + previous samples, so tok/s and latency are per-window."""

    def __init__(self) -> None:
        self.gen: deque[float] = deque([0.0], maxlen=_HISTORY)
        self.prompt: deque[float] = deque([0.0], maxlen=_HISTORY)
        self._prev_tokens: tuple[float, float, float] | None = None
        self._prev_ttft: tuple[float, float] | None = None
        self._prev_tpot: tuple[float, float] | None = None

    def rates(self, m: ServerMetrics, now: float) -> tuple[float | None, float | None]:
        counters = m.token_counters()
        gen_tps = prompt_tps = None
        if counters is not None:
            prompt, generation = counters
            if self._prev_tokens is not None:
                p0, g0, t0 = self._prev_tokens
                dt = now - t0
                if dt > 0 and prompt >= p0 and generation >= g0:
                    prompt_tps = (prompt - p0) / dt
                    gen_tps = (generation - g0) / dt
            self._prev_tokens = (prompt, generation, now)
        self.gen.append(gen_tps or 0.0)
        self.prompt.append(prompt_tps or 0.0)
        return gen_tps, prompt_tps

    def window_avg(self, cur_sum, cur_count, which) -> float | None:
        attr = "_prev_ttft" if which == "ttft" else "_prev_tpot"
        prev = getattr(self, attr)
        if cur_sum is None or cur_count is None:
            return None
        setattr(self, attr, (cur_sum, cur_count))
        if prev is None:
            return cur_sum / cur_count if cur_count > 0 else None
        d_sum, d_count = cur_sum - prev[0], cur_count - prev[1]
        if d_count > 0:
            return d_sum / d_count
        return cur_sum / cur_count if cur_count > 0 else None


def render_frame(row: DashboardRow, m: ServerMetrics | None, gpus, state: MonitorState) -> Group:
    lines: list[Text] = []
    title = f"[b]{row.profile_name}[/b]  " + (
        f"[{_VLLM_BLUE}]vLLM[/]" if row.backend == "vllm" else "[green]llama.cpp[/]"
    )
    if row.model:
        title += f"  {row.model}"
    if row.port:
        title += f"  :{row.port}"
    lines.append(Text.from_markup(title))
    lines.append(Text())

    if m is None:
        lines.append(Text.from_markup(t("[dim]server unreachable / metrics off[/]",
                                        "[dim]서버 응답 없음 / metrics 꺼짐[/]")))
    else:
        now = monotonic()
        gen, prompt = state.rates(m, now)
        lines.append(Text.from_markup(t("[b]Throughput[/b]  (tok/s)", "[b]처리량[/b]  (tok/s)")))
        gen_s = f"[b green]{gen:7.1f}[/]" if gen is not None else "[dim]     —  [/]"
        prompt_s = f"[b]{prompt:7.1f}[/]" if prompt is not None else "[dim]     —  [/]"
        lines.append(Text.from_markup(f"  generation {gen_s}  [green]{_sparkline(state.gen)}[/]"))
        lines.append(Text.from_markup(f"  prompt     {prompt_s}  [{_VLLM_BLUE}]{_sparkline(state.prompt)}[/]"))
        lines.append(Text())

        run = "—" if m.requests_running is None else str(int(m.requests_running))
        wait = "—" if m.requests_waiting is None else str(int(m.requests_waiting))
        wbar = _bar(min((m.requests_waiting or 0.0) / 8.0, 1.0), 16, "yellow") if m.requests_waiting is not None else ""
        lines.append(Text.from_markup(
            t(f"[b]Requests[/b]   running [b]{run}[/]   waiting [b]{wait}[/] {wbar}",
              f"[b]요청[/b]   실행 [b]{run}[/]   대기 [b]{wait}[/] {wbar}")))

        if m.kv_cache_usage is not None:
            lines.append(Text.from_markup(
                t(f"[b]KV cache[/b]   {m.kv_cache_usage * 100:4.0f}%  {_bar(m.kv_cache_usage)}",
                  f"[b]KV 캐시[/b]   {m.kv_cache_usage * 100:4.0f}%  {_bar(m.kv_cache_usage)}")))
        else:
            lines.append(Text.from_markup(t("[b]KV cache[/b]   [dim]—[/]", "[b]KV 캐시[/b]   [dim]—[/]")))

        ttft = state.window_avg(m.ttft_sum, m.ttft_count, "ttft")
        tpot = state.window_avg(m.tpot_sum, m.tpot_count, "tpot")
        ttft_s = f"{ttft:.2f}s" if ttft is not None else "—"
        tpot_s = f"{tpot * 1000:.0f} ms" if tpot is not None else "—"
        lines.append(Text.from_markup(
            t(f"[b]Latency[/b]    TTFT [b]{ttft_s}[/]   per-token [b]{tpot_s}[/]",
              f"[b]지연[/b]    TTFT [b]{ttft_s}[/]   토큰당 [b]{tpot_s}[/]")))

    lines.append(Text())
    ids = {p for p in row.gpu_id.split(",") if p} if row.gpu_id else set()
    mine = [g for g in gpus if not ids or g.index in ids] or gpus
    if mine:
        lines.append(Text.from_markup(t("[b]GPU[/b]", "[b]GPU[/b]")))
        for g in mine:
            try:
                util, used, total = float(g.utilization), float(g.memory_used), float(g.memory_total)
            except ValueError:
                continue
            power = f"  {g.power}W" if g.power and g.power not in ("[N/A]", "") else ""
            lines.append(Text.from_markup(
                f"  [b]GPU{g.index}[/b]  util {util:3.0f}% {_bar(util / 100, 18, 'green')}  "
                f"{used / 1024:.1f}/{total / 1024:.1f}GB {_bar(used / total if total else 0, 12, 'magenta')}  "
                f"{g.temperature}°C{power}"))
    lines.append(Text())
    lines.append(Text.from_markup(t("[dim]q: back · auto-refresh 1s[/]", "[dim]q: 뒤로 · 1초마다 갱신[/]")))
    return Group(*lines)


async def run_plain_monitor(row: DashboardRow, interval: float = 1.0) -> None:
    from rich.live import Live

    console = Console()
    state = MonitorState()
    old_attrs = None
    fd = None
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_attrs = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except Exception:
        old_attrs = None

    try:
        with Live(console=console, screen=True, auto_refresh=False) as live:
            while True:
                m = await fetch_server_metrics(row.port) if row.port else None
                gpus = await common_docker.get_gpu_info()
                live.update(render_frame(row, m, gpus, state))
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


async def _running_rows() -> list[DashboardRow]:
    from tui.backends.llamacpp.adapter import LlamacppAdapter
    from tui.backends.vllm.adapter import VllmAdapter

    try:
        running = await common_docker.running_container_names()
    except Exception:
        running = set()
    rows: list[DashboardRow] = []
    for adapter in (VllmAdapter(), LlamacppAdapter()):
        try:
            rows.extend(adapter.rows(running))
        except Exception:
            pass
    return [r for r in rows if r.running]


async def _resolve_and_run(profile: str | None) -> int:
    rows = await _running_rows()
    if not rows:
        print(t("No running models to monitor. Start one first (`llmux up <profile>`).",
                "모니터할 실행 중인 모델이 없습니다. 먼저 시작하세요 (`llmux up <profile>`)."))
        return 1
    if profile:
        match = next((r for r in rows if r.profile_name == profile), None)
        if match is None:
            print(t(f"'{profile}' is not running. Running: {', '.join(r.profile_name for r in rows)}",
                    f"'{profile}' 은 실행 중이 아닙니다. 실행 중: {', '.join(r.profile_name for r in rows)}"))
            return 1
        target = match
    elif len(rows) == 1:
        target = rows[0]
    else:
        names = ", ".join(r.profile_name for r in rows)
        print(t(f"Multiple models running — pass one: {names}",
                f"실행 중인 모델이 여러 개입니다 — 하나를 지정하세요: {names}"))
        return 1
    await run_plain_monitor(target)
    return 0


def run_cli(profile: str | None = None) -> int:
    """`llmux top [profile]` entry point."""
    try:
        return asyncio.run(_resolve_and_run(profile))
    except KeyboardInterrupt:
        return 0
