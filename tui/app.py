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
        Binding("q", "quit", "Quit", show=True),
        Binding("f1", "show_dashboard", "Dashboard", show=False),
        Binding("f2", "show_configs", "Configs", show=False),
        Binding("f3", "show_system", "System", show=False),
        Binding("question_mark", "toggle_help", "Help", show=False),
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
        if not isinstance(self.screen, DashboardScreen):
            self.switch_screen("dashboard")

    def action_show_configs(self) -> None:
        if not isinstance(self.screen, ConfigListScreen):
            self.switch_screen("configs")

    def action_show_system(self) -> None:
        if not isinstance(self.screen, SystemScreen):
            self.switch_screen("system")

    def action_toggle_help(self) -> None:
        self.notify(
            "[b]Dashboard[/b]\n"
            "  Enter  Open action menu for selected profile\n"
            "  w Quick Setup · n New Profile\n"
            "  s System Info · c Configs · r Refresh\n"
            "\n[b]Direct Shortcuts[/b] (power-user)\n"
            "  u Start · d Stop · l Logs\n"
            "  e Edit Profile · x Delete\n"
            "\n[b]Global[/b]\n"
            "  F1 Dashboard · F2 Configs · F3 System · q Quit",
            title="Keyboard Shortcuts",
            timeout=12,
        )


def main() -> None:
    app = VllmApp()
    app.run()


if __name__ == "__main__":
    main()
