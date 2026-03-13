"""vLLM Compose TUI - Main Application."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header

from tui.screens.dashboard import DashboardScreen
from tui.screens.config import ConfigListScreen
from tui.screens.system import SystemScreen


class VllmApp(App):
    """vLLM Container Management TUI."""

    TITLE = "vLLM Compose"
    SUB_TITLE = "Container Management"
    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True, priority=True),
        Binding("f1", "show_dashboard", "Dashboard"),
        Binding("f2", "show_configs", "Configs"),
        Binding("f3", "show_system", "System"),
        Binding("question_mark", "toggle_help", "Help"),
    ]

    SCREENS = {
        "dashboard": DashboardScreen,
        "configs": ConfigListScreen,
        "system": SystemScreen,
    }

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

    def on_mount(self) -> None:
        self.push_screen("dashboard")

    def action_show_dashboard(self) -> None:
        self.switch_screen("dashboard")

    def action_show_configs(self) -> None:
        self.push_screen("configs")

    def action_show_system(self) -> None:
        self.push_screen("system")

    def action_toggle_help(self) -> None:
        self.notify(
            "[b]Global Keys:[/b]\n"
            "  F1 = Dashboard\n"
            "  F2 = Configs\n"
            "  F3 = System Info\n"
            "  q  = Quit\n"
            "\n[b]Dashboard Keys:[/b]\n"
            "  u = Start  d = Stop  l = Logs\n"
            "  n = New Profile  e = Edit  r = Refresh\n"
            "  s = System Info",
            title="Help",
            timeout=8,
        )


def main() -> None:
    app = VllmApp()
    app.run()


if __name__ == "__main__":
    main()
