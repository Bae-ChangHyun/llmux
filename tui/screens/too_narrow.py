"""Width-guard screen — terminal이 MIN_WIDTH보다 좁아지면 표시되는 안내 화면.

nvitop 처럼, 좁은 폭에서는 form/modal 컨텐츠가 잘려 읽을 수 없으므로
앱 위에 덮어씌워 "터미널을 늘려 주세요" 안내만 띄운다. 폭이 회복되면
자동으로 dismiss 된다 (LlmuxApp.on_resize 가 push/pop 을 관리).
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Static


class TooNarrowScreen(Screen):
    """Modal-like overlay shown while the terminal is too narrow."""

    DEFAULT_CSS = """
    TooNarrowScreen {
        align: center middle;
        background: $surface;
    }

    TooNarrowScreen > Vertical {
        width: auto;
        height: auto;
        padding: 1 2;
        border: round $warning;
        background: $surface;
    }

    TooNarrowScreen #title {
        text-style: bold;
        color: $warning;
        width: 100%;
        content-align: center middle;
        margin-bottom: 1;
    }

    TooNarrowScreen .row {
        width: 100%;
        content-align: center middle;
        color: $text;
    }

    TooNarrowScreen .hint {
        color: $text-muted;
        margin-top: 1;
    }
    """

    def __init__(self, min_width: int, current_width: int) -> None:
        super().__init__()
        self._min_width = min_width
        self._current_width = current_width

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("⚠  Terminal too narrow", id="title")
            yield Static(self._status_line(), id="status", classes="row")
            yield Static(
                f"Please widen the terminal to at least {self._min_width} columns.",
                classes="hint",
            )

    def _status_line(self) -> str:
        return (
            f"current: [b]{self._current_width}[/b] cols   "
            f"required: [b]{self._min_width}[/b] cols"
        )

    def update_width(self, current_width: int) -> None:
        """LlmuxApp.on_resize 가 호출. 실시간으로 표시 폭을 갱신."""
        self._current_width = current_width
        try:
            self.query_one("#status", Static).update(self._status_line())
        except Exception:
            pass
