"""ProfileActionScreen — Enter-on-profile 의 액션 메뉴 (통합 Dashboard 가 push)."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from tui.common.i18n import t


class ProfileActionScreen(ModalScreen[str]):
    """Context action menu for a selected profile.

    Shows relevant actions based on container state (running/stopped).
    Returns the action id string on selection, empty string on cancel.
    """

    BINDINGS = [Binding("escape", "cancel", show=False)]

    DEFAULT_CSS = """
    ProfileActionScreen {
        align: center middle;
    }
    ProfileActionScreen > Vertical {
        background: $surface;
        border: round $primary;
        padding: 1 2;
        width: 42;
        height: auto;
    }
    ProfileActionScreen #action-title {
        text-style: bold;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }
    ProfileActionScreen OptionList {
        height: auto;
        max-height: 14;
    }
    """

    def __init__(self, profile_name: str, running: bool) -> None:
        super().__init__()
        self.profile_name = profile_name
        self._profile_running = running

    def compose(self) -> ComposeResult:
        if self._profile_running:
            status = t("[green]● running[/]", "[green]● 실행 중[/]")
        else:
            status = t("[dim]○ stopped[/]", "[dim]○ 중지됨[/]")

        options: list[Option] = []
        if self._profile_running:
            options.append(Option(t("■ Stop Container", "■ 컨테이너 중지"), id="stop"))
        else:
            options.append(Option(t("▶ Start Container", "▶ 컨테이너 시작"), id="start"))
            options.append(
                Option(t("⬇ Prepare (download only)", "⬇ 준비 (다운로드만)"), id="prepare")
            )
        # `docker logs` also works on exited containers, so the log viewer is
        # offered in both states — matching the `l` binding on the dashboard.
        options.append(Option(t("◉ View Logs", "◉ 로그 보기"), id="logs"))
        if self._profile_running:
            options.append(Option(t("📊 Monitor", "📊 모니터"), id="monitor"))
            options.append(Option(t("⚡ Benchmark", "⚡ 벤치마크"), id="benchmark"))
        options.append(Option(t("✎ Edit Profile", "✎ 프로필 편집"), id="edit_profile"))
        options.append(Option(t("⚙ Edit Config", "⚙ Config 편집"), id="edit_config"))
        if not self._profile_running:
            options.append(Option(t("✗ Delete Profile", "✗ 프로필 삭제"), id="delete"))

        with Vertical():
            yield Static(
                f"{self.profile_name}  {status}", id="action-title"
            )
            yield OptionList(*options, id="action-list")

    @on(OptionList.OptionSelected, "#action-list")
    def _on_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id or "")

    def action_cancel(self) -> None:
        self.dismiss("")
