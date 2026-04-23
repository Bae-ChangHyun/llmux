"""ActionModal + LogViewer — 통합 Dashboard 가 llama.cpp 프로필에 push.

구 `DashboardScreen` 은 `tui/screens/dashboard.py` 의 통합판으로 대체되어 제거됨.
"""

from __future__ import annotations

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Label,
    OptionList,
    RichLog,
    Static,
)
from textual.widgets.option_list import Option

from tui.backends.llamacpp import backend


class ActionModal(ModalScreen[str]):
    """프로필 액션 선택 모달. vLLM ProfileActionScreen 과 동일한 OptionList 스타일."""

    BINDINGS = [Binding("escape", "cancel", show=False)]

    DEFAULT_CSS = """
    ActionModal {
        align: center middle;
    }
    ActionModal > Vertical {
        background: $surface;
        border: round $primary;
        padding: 1 2;
        width: 42;
        height: auto;
    }
    ActionModal #action-title {
        text-style: bold;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }
    ActionModal OptionList {
        height: auto;
        max-height: 14;
    }
    """

    def __init__(self, profile: backend.Profile) -> None:
        super().__init__()
        self.profile = profile

    def compose(self) -> ComposeResult:
        p = self.profile
        running = p.running

        if running:
            status = "[green]● running[/]"
        else:
            status = "[dim]○ stopped[/]"

        options: list[Option] = []
        if running:
            options.append(Option("■ Stop Container", id="stop"))
            options.append(Option("◉ View Logs", id="logs"))
            options.append(Option("⚡ Benchmark", id="benchmark"))
        else:
            options.append(Option("▶ Start Container", id="start"))
        options.append(Option("✎ Edit Profile", id="edit-profile"))
        options.append(Option("⚙ Edit Config", id="edit-config"))
        if not running:
            options.append(Option("✗ Delete Profile", id="delete-profile"))

        with Vertical():
            yield Static(f"{p.name}  {status}", id="action-title")
            yield OptionList(*options, id="action-list")

    @on(OptionList.OptionSelected, "#action-list")
    def _on_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id or "")

    def action_cancel(self) -> None:
        self.dismiss("")


class LogViewer(ModalScreen[None]):
    """docker logs -f 실시간 표시. RichLog 로 스크롤/자동 follow 지원."""

    BINDINGS = [
        Binding("escape,q", "close", "Close", show=True),
        Binding("f", "toggle_follow", "Follow on/off"),
    ]

    def __init__(self, container_name: str) -> None:
        super().__init__()
        self.container_name = container_name

    def compose(self) -> ComposeResult:
        with Vertical(id="log-container"):
            yield Label(
                f"  Logs — {self.container_name}   "
                "[dim](esc:close  f:auto-follow  ↑↓/PgUp/PgDn:scroll)[/dim]",
                id="log-title",
            )
            yield RichLog(
                id="log-body",
                highlight=False,
                markup=False,
                wrap=False,
                auto_scroll=True,
                max_lines=5000,
            )

    def on_mount(self) -> None:
        self._stream()

    @work(exclusive=True, group="logviewer-stream")
    async def _stream(self) -> None:
        log = self.query_one("#log-body", RichLog)
        try:
            async for line in backend.stream_logs(self.container_name, lines=200):
                log.write(backend.strip_ansi(line))
        except Exception as exc:  # pragma: no cover
            log.write(f"[로그 스트림 오류] {exc}")

    def action_toggle_follow(self) -> None:
        log = self.query_one("#log-body", RichLog)
        log.auto_scroll = not log.auto_scroll
        if log.auto_scroll:
            log.scroll_end(animate=False)
        self.notify(f"auto-follow: {'ON' if log.auto_scroll else 'OFF'}", timeout=2)

    def action_close(self) -> None:
        self.workers.cancel_group(self, "logviewer-stream")
        self.dismiss(None)


class StartScreen(Screen[None]):
    """Full-screen llama.cpp startup log, then live container logs on success."""

    BINDINGS = [
        Binding("escape", "close", "Back"),
        Binding("q", "close", "Back"),
        Binding("f", "toggle_follow", "Follow on/off"),
    ]

    DEFAULT_CSS = """
    StartScreen {
        layout: vertical;
    }
    StartScreen #startup-title {
        dock: top;
        height: 1;
        color: $text-muted;
        text-style: bold;
        padding: 0 2;
        margin: 1 0;
    }
    StartScreen RichLog {
        height: 1fr;
        margin: 0;
    }
    """

    def __init__(self, profile_name: str) -> None:
        super().__init__()
        self.profile_name = profile_name
        self._profile = backend.load_profile(profile_name)

    def compose(self) -> ComposeResult:
        yield Label(
            f"Starting llama.cpp: {self.profile_name}  "
            "[dim](q/Esc:back  f:auto-follow  ↑↓/PgUp/PgDn:scroll)[/dim]",
            id="startup-title",
        )
        yield RichLog(
            id="startup-log",
            highlight=False,
            markup=False,
            wrap=False,
            auto_scroll=True,
            max_lines=5000,
        )

    def on_mount(self) -> None:
        self._start()

    @work(exclusive=True, group="llamacpp-start")
    async def _start(self) -> None:
        log = self.query_one("#startup-log", RichLog)
        code = -1
        async for msg_type, data in backend.stream_script("switch.sh", self.profile_name):
            if msg_type == "log":
                log.write(backend.strip_ansi(data))
            elif msg_type == "rc":
                code = int(data)
        if code != 0:
            log.write(f"Failed to start (rc={code})")
            self.notify(
                f"llama.cpp start failed: {self.profile_name}", severity="error"
            )
            return

        self.notify(
            f"llama.cpp started: {self.profile_name} on {self._profile.endpoint}",
            timeout=6,
        )
        log.write("")
        log.write("Container started. Streaming logs...")
        try:
            async for line in backend.stream_logs(
                self._profile.container_name, lines=200
            ):
                log.write(backend.strip_ansi(line))
        except Exception as exc:  # pragma: no cover
            log.write(f"Log stream error: {exc}")

    def action_toggle_follow(self) -> None:
        log = self.query_one("#startup-log", RichLog)
        log.auto_scroll = not log.auto_scroll
        if log.auto_scroll:
            log.scroll_end(animate=False)
        self.notify(f"auto-follow: {'ON' if log.auto_scroll else 'OFF'}", timeout=2)

    def action_close(self) -> None:
        self.workers.cancel_group(self, "llamacpp-start")
        self.app.pop_screen()
