"""llm-compose — unified TUI for vLLM + llama.cpp."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header

from tui.backends.llamacpp.screens.config import ConfigListScreen as LlamacppConfigListScreen
from tui.backends.llamacpp.screens.system import SystemScreen as LlamacppSystemScreen
from tui.backends.vllm.screens.config import ConfigListScreen as VllmConfigListScreen
from tui.backends.vllm.screens.system import SystemScreen as VllmSystemScreen
from tui.screens.dashboard import DashboardScreen


class LlmComposeApp(App):
    """Unified vLLM + llama.cpp launcher."""

    TITLE = "llm-compose"
    SUB_TITLE = "vLLM + llama.cpp"
    CSS_PATH = "common/app.tcss"

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

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

    def on_mount(self) -> None:
        self.push_screen("dashboard")

    def action_show_dashboard(self) -> None:
        if not isinstance(self.screen, DashboardScreen):
            self.switch_screen("dashboard")

    def action_help(self) -> None:
        self.notify(
            "[b]Dashboard[/b]\n"
            "  Enter action menu · u/d/l start/stop/logs\n"
            "  e/c/x edit profile/config, delete\n"
            "  n new · s system · r refresh · q quit",
            title="llm-compose",
            timeout=10,
        )


def main() -> None:
    LlmComposeApp().run()


if __name__ == "__main__":
    main()
