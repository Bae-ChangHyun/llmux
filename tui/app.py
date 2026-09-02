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
from tui.common.i18n import t
from tui.screens.dashboard import DashboardScreen
from tui.screens.too_narrow import TooNarrowScreen


class LlmuxApp(App):
    """Unified vLLM + llama.cpp launcher."""

    TITLE = "llmux"
    SUB_TITLE = "vLLM + llama.cpp"
    CSS_PATH = "common/app.tcss"

    # Modals are width:65 + padding/border/margins; below 80 cols they clip.
    MIN_WIDTH = 80

    BINDINGS = [
        Binding("q", "quit", t("Quit", "종료"), show=True),
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
        self._width_guard_armed = False
        self._modal_generation = 0

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

    def on_mount(self) -> None:
        self.push_screen("dashboard")
        # Textual sends the initial Resize before on_mount, so arm after dashboard push.
        self._width_guard_armed = True
        self._enforce_width(self.size.width)

    @property
    def modal_generation(self) -> int:
        return self._modal_generation

    def can_push_modal(self, generation: int) -> bool:
        return (
            generation == self._modal_generation
            and self._too_narrow is None
            and self.size.width >= self.MIN_WIDTH
        )

    def on_resize(self, event: Resize) -> None:
        if not self._width_guard_armed:
            return
        self._enforce_width(event.size.width)

    def _enforce_width(self, width: int) -> None:
        if width < self.MIN_WIDTH:
            if self._too_narrow is None:
                self._modal_generation += 1
                self._too_narrow = TooNarrowScreen(self.MIN_WIDTH, width)
                self.push_screen(self._too_narrow)
            else:
                self._too_narrow.update_width(width)
            return
        if self._too_narrow is not None:
            guard = self._too_narrow
            if self.screen is guard:
                self._release_width_guard(guard)

    def _release_width_guard(self, guard: TooNarrowScreen) -> None:
        if self._too_narrow is not guard:
            return
        self._too_narrow = None
        self._modal_generation += 1
        if self.screen is guard:
            self.pop_screen()

    def action_show_dashboard(self) -> None:
        while not isinstance(self.screen, DashboardScreen) and len(self.screen_stack) > 1:
            if self.screen is self._too_narrow:
                if self.size.width < self.MIN_WIDTH:
                    return
                self._release_width_guard(self.screen)
                continue
            self.pop_screen()

    def action_help(self) -> None:
        self.notify(
            t(
                "[b]Dashboard[/b]\n"
                "  Enter action menu · u/d/l start/stop/logs\n"
                "  e/c/x edit profile/config, delete\n"
                "  m estimate model memory\n"
                "  C config list · n new · s system · r refresh · q quit",
                "[b]대시보드[/b]\n"
                "  Enter 작업 메뉴 · u/d/l 시작/중지/로그\n"
                "  e/c/x 프로필/config 편집, 삭제\n"
                "  m 모델 메모리 추정\n"
                "  C config 목록 · n 새로 · s 시스템 · r 새로고침 · q 종료",
            ),
            title="llmux",
            timeout=10,
        )


def main() -> None:
    """Entrypoint dispatching to CLI (typer) or TUI."""
    from tui.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
