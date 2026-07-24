"""Live btop-style monitor inside the Textual TUI (`v` on a dashboard row).

Draws the same `monitor_render.render_dashboard` view as `llmux top`: every GPU
always, plus a panel per running model. Opening it never requires a running
container — with nothing up you still get the GPU panel. Keys: `q`/Esc back,
`p` pause, `r` reset peaks, `+`/`-` interval, `l` language.
"""

from __future__ import annotations

import os
from time import monotonic

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from tui.common import docker as common_docker
from tui.common.adapter import DashboardRow
from tui.common.i18n import t
from tui.common.monitor_render import MonitorState, render_dashboard
from tui.common.plain_monitor import sample_entries

_MIN_INTERVAL = 0.25
_MAX_INTERVAL = 5.0


class MonitorScreen(Screen):
    BINDINGS = [
        Binding("q", "go_back", t("Back", "뒤로")),
        Binding("escape", "go_back", show=False),
        Binding("p", "pause", t("Pause", "일시정지")),
        Binding("r", "reset_peaks", t("Reset", "초기화")),
        Binding("plus", "faster", t("Interval", "주기"), key_display="+/-"),
        Binding("minus", "slower", show=False),
        Binding("equals_sign", "faster", show=False),
        Binding("l", "toggle_lang", t("Lang", "언어")),
    ]

    DEFAULT_CSS = """
    #mon-scroll { padding: 0 1; }
    """

    def __init__(self, row: DashboardRow | None = None) -> None:
        super().__init__()
        # Focus the row it was opened from when that model is up; otherwise the
        # view is system-wide (every running model, or none).
        self._focus = row.profile_name if row is not None and row.running else None
        self._states: dict[str, MonitorState] = {}
        self._started = monotonic()
        self._interval = 1.0
        self._paused = False
        self._polling = False
        self._timer = None
        self._rendered = None
        self._last = {"entries": [], "gpus": [], "pcie": {}, "lag": 0.0, "ready": False}

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="mon-scroll"):
            yield Static("", id="mon")
        yield Footer()

    def on_mount(self) -> None:
        self._timer = self.set_interval(self._interval, self._poll)
        self.call_after_refresh(self._poll)

    async def _poll(self) -> None:
        # Serialized by set_interval; the guard covers an interval shorter than
        # one fetch so two polls never mutate state concurrently.
        if self._paused or self._polling:
            return
        self._polling = True
        try:
            t0 = monotonic()
            gpus = await common_docker.get_gpu_info()
            pcie = await common_docker.get_pcie_stats()
            lag_ms = (monotonic() - t0) * 1000
            entries = await sample_entries(self._focus, self._states, monotonic(), lag_ms)
            self._last.update(entries=entries, gpus=gpus, pcie=pcie, lag=lag_ms, ready=True)
            self._repaint()
        finally:
            self._polling = False

    def _repaint(self) -> None:
        if not self._last["ready"]:
            return
        self._rendered = render_dashboard(
            self._last["entries"], self._last["gpus"], self._last["pcie"],
            self.size.width or 100, paused=self._paused, interval=self._interval,
            uptime=monotonic() - self._started, lag_ms=self._last["lag"],
        )
        self.query_one("#mon", Static).update(self._rendered)

    def _restart_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
        self._timer = self.set_interval(self._interval, self._poll)

    def action_pause(self) -> None:
        self._paused = not self._paused
        self._repaint()

    def action_reset_peaks(self) -> None:
        for state in self._states.values():
            state.reset_peaks()
        self._repaint()

    def action_faster(self) -> None:
        self._interval = max(_MIN_INTERVAL, round(self._interval - 0.25, 2))
        self._restart_timer()
        self._repaint()

    def action_slower(self) -> None:
        self._interval = min(_MAX_INTERVAL, round(self._interval + 0.25, 2))
        self._restart_timer()
        self._repaint()

    def action_toggle_lang(self) -> None:
        os.environ["LLMUX_LANG"] = "en" if os.environ.get("LLMUX_LANG") == "ko" else "ko"
        self._repaint()

    def action_go_back(self) -> None:
        self.dismiss()
