"""ProfileActionScreen — Enter-on-profile 의 액션 메뉴 (통합 Dashboard 가 push).

구 `DashboardScreen` 은 `tui/screens/dashboard.py` 의 통합판으로 대체되어 제거됨.
"""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option


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
            status = "[green]● running[/]"
        else:
            status = "[dim]○ stopped[/]"

        options: list[Option] = []
        if self._profile_running:
            options.append(Option("■ Stop Container", id="stop"))
            options.append(Option("◉ View Logs", id="logs"))
            options.append(Option("⚡ Benchmark", id="benchmark"))
        else:
            options.append(Option("▶ Start Container", id="start"))
        options.append(Option("✎ Edit Profile", id="edit_profile"))
        options.append(Option("⚙ Edit Config", id="edit_config"))
        options.append(Option("✗ Delete Profile", id="delete"))

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
