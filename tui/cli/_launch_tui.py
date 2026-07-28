"""TUI launch helper, isolated so CLI subcommand modules don't import textual."""

from __future__ import annotations


def launch_tui() -> None:
    from tui.app import LlmuxApp

    LlmuxApp().run()
