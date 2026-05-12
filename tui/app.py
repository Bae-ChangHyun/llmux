"""llmux — unified TUI for vLLM + llama.cpp."""

from __future__ import annotations

import sys

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header

from tui.backends.llamacpp.screens.config import ConfigListScreen as LlamacppConfigListScreen
from tui.backends.llamacpp.screens.system import SystemScreen as LlamacppSystemScreen
from tui.backends.vllm.screens.config import ConfigListScreen as VllmConfigListScreen
from tui.backends.vllm.screens.system import SystemScreen as VllmSystemScreen
from tui.common.profile_store import PROJECT_ROOT
from tui.screens.dashboard import DashboardScreen


class LlmuxApp(App):
    """Unified vLLM + llama.cpp launcher."""

    TITLE = "llmux"
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
            title="llmux",
            timeout=10,
        )


def _maybe_show_shortcut_notice() -> None:
    """On first run, suggest adding a shell function so `llmux` works anywhere."""
    marker = PROJECT_ROOT / ".runtime" / ".shortcut-shown"
    if marker.exists():
        return

    func_line = f'llmux() {{ (cd "{PROJECT_ROOT}" && uv run llmux "$@"); }}'

    print(
        "\nTip: to run `llmux` from any directory, copy & run ONE of these (pick your shell):\n\n"
        f"  bash:  echo '{func_line}' >> ~/.bashrc && source ~/.bashrc\n"
        f"  zsh :  echo '{func_line}' >> ~/.zshrc  && source ~/.zshrc\n"
    )

    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    except OSError:
        pass

    try:
        input("\nPress Enter to continue… ")
    except (EOFError, KeyboardInterrupt):
        sys.exit(0)


def main() -> None:
    _maybe_show_shortcut_notice()
    LlmuxApp().run()


if __name__ == "__main__":
    main()
