"""Terminal width guard."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Static


class TooNarrowScreen(Screen):
    DEFAULT_CSS = """
    TooNarrowScreen {
        align: center middle;
        background: $surface;
    }

    /* width:1fr avoids Textual's circular auto/100% child constraint. */
    TooNarrowScreen > Vertical {
        width: 1fr;
        max-width: 54;
        height: auto;
        margin: 1 1;
        padding: 1 2;
        border: round $warning;
        background: $surface;
    }

    TooNarrowScreen #title {
        text-style: bold;
        color: $warning;
        width: 100%;
        text-align: center;
        margin-bottom: 1;
    }

    TooNarrowScreen .row {
        width: 100%;
        text-align: center;
        color: $text;
    }

    TooNarrowScreen .hint {
        width: 100%;
        text-align: center;
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
        self._current_width = current_width
        try:
            self.query_one("#status", Static).update(self._status_line())
        except Exception:
            pass

    def on_screen_resume(self) -> None:
        if self.app.size.width < self._min_width:
            self.update_width(self.app.size.width)
            return
        release = getattr(self.app, "_release_width_guard", None)
        if release is not None:
            release(self)
