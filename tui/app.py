"""llmux — unified TUI for vLLM + llama.cpp."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.events import Resize
from textual.widgets import Footer, Header

from tui.backends.llamacpp.screens.config import ConfigListScreen as LlamacppConfigListScreen
from tui.backends.llamacpp.screens.system import SystemScreen as LlamacppSystemScreen
from tui.backends.vllm.screens.config import ConfigListScreen as VllmConfigListScreen
from tui.backends.vllm.screens.system import SystemScreen as VllmSystemScreen
from tui.screens.dashboard import DashboardScreen
from tui.screens.too_narrow import TooNarrowScreen


class LlmuxApp(App):
    """Unified vLLM + llama.cpp launcher."""

    TITLE = "llmux"
    SUB_TITLE = "vLLM + llama.cpp"
    CSS_PATH = "common/app.tcss"

    MIN_WIDTH = 80
    """Below this terminal width every modal/form gets clipped horizontally
    (existing modals are width:65 + padding/border + outer margins). When the
    terminal is narrower we push TooNarrowScreen as a full-screen guard and
    pop it back as soon as the user widens the terminal."""

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("f1", "show_dashboard", "Dashboard", show=False),
        Binding("question_mark", "help", "Help", show=False),
    ]

    SCREENS = {
        "dashboard": DashboardScreen,
        "vllm_configs": VllmConfigListScreen,
        "vllm_system": VllmSystemScreen,
        "llamacpp_configs": LlamacppConfigListScreen,
        "llamacpp_system": LlamacppSystemScreen,
    }

    def __init__(self) -> None:
        super().__init__()
        self._too_narrow: TooNarrowScreen | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

    def on_mount(self) -> None:
        self.push_screen("dashboard")
        self._enforce_width(self.size.width)

    def on_resize(self, event: Resize) -> None:
        self._enforce_width(event.size.width)

    def _enforce_width(self, width: int) -> None:
        if width < self.MIN_WIDTH:
            if self._too_narrow is None:
                self._too_narrow = TooNarrowScreen(self.MIN_WIDTH, width)
                self.push_screen(self._too_narrow)
            else:
                self._too_narrow.update_width(width)
            return
        if self._too_narrow is not None:
            guard = self._too_narrow
            self._too_narrow = None
            if self.screen is guard:
                self.pop_screen()

    def action_show_dashboard(self) -> None:
        if not isinstance(self.screen, DashboardScreen):
            self.switch_screen("dashboard")

    def action_help(self) -> None:
        self.notify(
            "[b]Dashboard[/b]\n"
            "  Enter action menu · u/d/l start/stop/logs\n"
            "  e/c/x edit profile/config, delete\n"
            "  n new · s system · r refresh · q quit",
            title="llmux",
            timeout=10,
        )


def main() -> None:
    """Entrypoint dispatching to CLI (typer) or TUI."""
    from tui.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
