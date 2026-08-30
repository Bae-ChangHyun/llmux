"""ActionModal + LogViewer — 통합 Dashboard 가 llama.cpp 프로필에 push."""

from __future__ import annotations

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Label,
    OptionList,
    RichLog,
    Static,
)
from textual.widgets.option_list import Option

from tui.backends.llamacpp import backend
from tui.common.i18n import t


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
            status = t("[green]● running[/]", "[green]● 실행 중[/]")
        else:
            status = t("[dim]○ stopped[/]", "[dim]○ 중지됨[/]")

        options: list[Option] = []
        if running:
            options.append(Option(t("■ Stop Container", "■ 컨테이너 중지"), id="stop"))
        else:
            options.append(Option(t("▶ Start Container", "▶ 컨테이너 시작"), id="start"))
            options.append(
                Option(t("⬇ Prepare (download only)", "⬇ 준비 (다운로드만)"), id="prepare")
            )
        # `docker logs` also works on exited containers, so the log viewer is
        # offered in both states — matching the `l` binding on the dashboard.
        options.append(Option(t("◉ View Logs", "◉ 로그 보기"), id="logs"))
        if running:
            options.append(Option(t("📊 Monitor", "📊 모니터"), id="monitor"))
            options.append(Option(t("⚡ Benchmark", "⚡ 벤치마크"), id="benchmark"))
        options.append(Option(t("✎ Edit Profile", "✎ 프로필 편집"), id="edit-profile"))
        options.append(Option(t("⧉ Clone Profile", "⧉ 프로필 복제"), id="clone-profile"))
        options.append(Option(t("↻ Render Runtime Env", "↻ 런타임 환경 렌더링"), id="render-env"))
        options.append(Option(t("⚙ Edit Config", "⚙ Config 편집"), id="edit-config"))
        if not running:
            options.append(Option(t("✗ Delete Profile", "✗ 프로필 삭제"), id="delete-profile"))

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
        Binding("escape,q", "close", t("Close", "닫기"), show=True),
        Binding("f", "toggle_follow", t("Follow on/off", "자동 스크롤")),
    ]

    def __init__(self, container_name: str) -> None:
        super().__init__()
        self.container_name = container_name

    def compose(self) -> ComposeResult:
        with Vertical(id="log-container"):
            yield Label(
                t(
                    f"  Logs — {self.container_name}   "
                    "[dim](esc:close  f:auto-follow  ↑↓/PgUp/PgDn:scroll)[/dim]",
                    f"  로그 — {self.container_name}   "
                    "[dim](esc:닫기  f:자동 추적  ↑↓/PgUp/PgDn:스크롤)[/dim]",
                ),
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
            async for line in backend.stream_logs(self.container_name, tail=200):
                log.write(backend.strip_ansi(line))
        except Exception as exc:  # pragma: no cover
            log.write(t(f"[log stream error] {exc}", f"[로그 스트림 오류] {exc}"))

    def action_toggle_follow(self) -> None:
        log = self.query_one("#log-body", RichLog)
        log.auto_scroll = not log.auto_scroll
        if log.auto_scroll:
            log.scroll_end(animate=False)
        self.notify(
            t(
                f"auto-follow: {'ON' if log.auto_scroll else 'OFF'}",
                f"자동 추적: {'켜짐' if log.auto_scroll else '꺼짐'}",
            ),
            timeout=2,
        )

    def action_close(self) -> None:
        self.workers.cancel_group(self, "logviewer-stream")
        self.dismiss(None)
