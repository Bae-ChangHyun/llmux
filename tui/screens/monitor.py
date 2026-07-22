"""Live single-model monitor — a btop-style detail view for one running profile.

Pushed from the dashboard on a running row. Polls the server's `/metrics` and
`nvidia-smi` once a second and renders throughput sparklines, request queue,
KV-cache usage, latency, and per-GPU bars. TUI-only by design; the scriptable
equivalent is `llmux stats --json`.
"""

from __future__ import annotations

from collections import deque
from time import monotonic

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Sparkline, Static

from tui.common import docker as common_docker
from tui.common.adapter import DashboardRow
from tui.common.i18n import t
from tui.common.metrics import ServerMetrics, fetch_server_metrics

_HISTORY = 48


def _bar(ratio: float | None, width: int = 24, color: str = "cyan") -> str:
    if ratio is None:
        return "[dim]" + "░" * width + "[/]"
    ratio = max(0.0, min(1.0, ratio))
    filled = int(round(ratio * width))
    return f"[{color}]{'█' * filled}[/][dim]{'░' * (width - filled)}[/]"


class MonitorScreen(Screen):
    BINDINGS = [
        Binding("escape,q", "go_back", t("Back", "뒤로")),
    ]

    def __init__(self, row: DashboardRow) -> None:
        super().__init__()
        self._row = row
        self._gen_hist: deque[float] = deque([0.0], maxlen=_HISTORY)
        self._prompt_hist: deque[float] = deque([0.0], maxlen=_HISTORY)
        self._prev_tokens: tuple[float, float, float] | None = None
        self._prev_ttft: tuple[float, float] | None = None
        self._prev_tpot: tuple[float, float] | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="monitor-body"):
            yield Static(id="mon-title")
            yield Static(t("[b]Throughput[/b]  (tok/s)", "[b]처리량[/b]  (tok/s)"), classes="mon-h")
            yield Static(id="mon-gen-label")
            yield Sparkline([0.0], id="mon-gen-spark")
            yield Static(id="mon-prompt-label")
            yield Sparkline([0.0], id="mon-prompt-spark")
            yield Static(id="mon-requests", classes="mon-block")
            yield Static(id="mon-kv", classes="mon-block")
            yield Static(id="mon-latency", classes="mon-block")
            yield Static(t("[b]GPU[/b]", "[b]GPU[/b]"), classes="mon-h")
            yield Static(id="mon-gpu")
        yield Footer()

    DEFAULT_CSS = """
    #monitor-body { padding: 1 2; }
    .mon-h { margin-top: 1; color: $accent; }
    .mon-block { margin-top: 1; }
    #mon-gen-spark { height: 3; color: $success; }
    #mon-prompt-spark { height: 3; color: $primary; }
    """

    def on_mount(self) -> None:
        title = f"{self._row.profile_name}  ·  {self._row.backend}"
        if self._row.model:
            title += f"  ·  {self._row.model}"
        if self._row.port:
            title += f"  ·  :{self._row.port}"
        self.query_one("#mon-title", Static).update(f"[b]{title}[/b]")
        self._poll()
        self._timer = self.set_interval(1.0, lambda: self._poll())

    @work(exclusive=True, group="monitor-poll")
    async def _poll(self) -> None:
        if self._row.port is None:
            self._set_unreachable(t("no port", "포트 없음"))
            return
        metrics = await fetch_server_metrics(self._row.port)
        gpus = await common_docker.get_gpu_info()
        if metrics is None:
            self._set_unreachable(t("server unreachable / metrics off", "서버 응답 없음 / metrics 꺼짐"))
        else:
            self._render_metrics(metrics)
        self._render_gpu(gpus)

    def _set_unreachable(self, why: str) -> None:
        self.query_one("#mon-gen-label", Static).update(t(f"[dim]{why}[/]", f"[dim]{why}[/]"))
        for wid in ("#mon-prompt-label", "#mon-requests", "#mon-kv", "#mon-latency"):
            self.query_one(wid, Static).update("")

    def _render_metrics(self, m: ServerMetrics) -> None:
        now = monotonic()
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
        self._gen_hist.append(gen_tps or 0.0)
        self._prompt_hist.append(prompt_tps or 0.0)
        self.query_one("#mon-gen-spark", Sparkline).data = list(self._gen_hist)
        self.query_one("#mon-prompt-spark", Sparkline).data = list(self._prompt_hist)
        self.query_one("#mon-gen-label", Static).update(
            t(f"generation  [b green]{gen_tps:6.1f}[/] tok/s" if gen_tps is not None else "generation  [dim]  —  [/]",
              f"생성  [b green]{gen_tps:6.1f}[/] tok/s" if gen_tps is not None else "생성  [dim]  —  [/]")
        )
        self.query_one("#mon-prompt-label", Static).update(
            t(f"prompt      [b]{prompt_tps:6.1f}[/] tok/s" if prompt_tps is not None else "prompt      [dim]  —  [/]",
              f"프롬프트  [b]{prompt_tps:6.1f}[/] tok/s" if prompt_tps is not None else "프롬프트  [dim]  —  [/]")
        )

        run = m.requests_running
        wait = m.requests_waiting
        run_s = f"{int(run)}" if run is not None else "—"
        wait_s = f"{int(wait)}" if wait is not None else "—"
        wait_bar = _bar(min((wait or 0.0) / 8.0, 1.0), width=16, color="yellow") if wait is not None else ""
        self.query_one("#mon-requests", Static).update(
            t(f"[b]Requests[/b]   running [b]{run_s}[/]   waiting [b]{wait_s}[/] {wait_bar}",
              f"[b]요청[/b]   실행 [b]{run_s}[/]   대기 [b]{wait_s}[/] {wait_bar}")
        )

        kv = m.kv_cache_usage
        if kv is not None:
            self.query_one("#mon-kv", Static).update(
                t(f"[b]KV cache[/b]   {kv * 100:4.0f}%  {_bar(kv)}",
                  f"[b]KV 캐시[/b]   {kv * 100:4.0f}%  {_bar(kv)}")
            )
        else:
            self.query_one("#mon-kv", Static).update(t("[b]KV cache[/b]   [dim]—[/]", "[b]KV 캐시[/b]   [dim]—[/]"))

        ttft = self._window_avg(m.ttft_sum, m.ttft_count, "_prev_ttft")
        tpot = self._window_avg(m.tpot_sum, m.tpot_count, "_prev_tpot")
        ttft_s = f"{ttft:.2f}s" if ttft is not None else "—"
        tpot_s = f"{tpot * 1000:.0f} ms" if tpot is not None else "—"
        self.query_one("#mon-latency", Static).update(
            t(f"[b]Latency[/b]    TTFT [b]{ttft_s}[/]   per-token [b]{tpot_s}[/]",
              f"[b]지연[/b]    TTFT [b]{ttft_s}[/]   토큰당 [b]{tpot_s}[/]")
        )

    def _window_avg(self, cur_sum: float | None, cur_count: float | None, attr: str) -> float | None:
        prev = getattr(self, attr)
        if cur_sum is None or cur_count is None:
            return None
        setattr(self, attr, (cur_sum, cur_count))
        if prev is None:
            return cur_sum / cur_count if cur_count > 0 else None
        d_sum = cur_sum - prev[0]
        d_count = cur_count - prev[1]
        if d_count > 0:
            return d_sum / d_count
        return cur_sum / cur_count if cur_count > 0 else None

    def _render_gpu(self, gpus: list[common_docker.GpuInfo]) -> None:
        ids = {p for p in self._row.gpu_id.split(",") if p} if self._row.gpu_id else set()
        mine = [g for g in gpus if not ids or g.index in ids] or gpus
        lines: list[str] = []
        for g in mine:
            try:
                util = float(g.utilization)
                used = float(g.memory_used)
                total = float(g.memory_total)
            except ValueError:
                continue
            mem_ratio = used / total if total else 0.0
            power = f"  {g.power}W" if g.power and g.power not in ("[N/A]", "") else ""
            lines.append(
                f"[b]GPU{g.index}[/b]  util {util:3.0f}% {_bar(util / 100, 18, 'green')}  "
                f"{used / 1024:.1f}/{total / 1024:.1f}GB {_bar(mem_ratio, 12, 'magenta')}  "
                f"{g.temperature}°C{power}"
            )
        self.query_one("#mon-gpu", Static).update(
            "\n".join(lines) if lines else t("[dim]GPU info unavailable[/]", "[dim]GPU 정보 없음[/]")
        )

    def action_go_back(self) -> None:
        self.dismiss()
