"""Container management screens: up, down, and log viewer."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, Center
from textual.screen import Screen, ModalScreen
from textual.widgets import (
    Button,
    Static,
    Label,
    Input,
    RadioSet,
    RadioButton,
    RichLog,
    Header,
    LoadingIndicator,
)
from textual import work, on

from tui.backend import (
    container_up,
    container_down,
    stream_container_logs,
    load_profile,
)


# ---------------------------------------------------------------------------
# Version option constants
# ---------------------------------------------------------------------------

VERSION_LOCAL_LATEST = "Local Latest"
VERSION_OFFICIAL = "Official"
VERSION_NIGHTLY = "Nightly"
VERSION_DEV_BUILD = "Dev Build"
VERSION_CUSTOM_TAG = "Custom Tag"


# ---------------------------------------------------------------------------
# ContainerUpScreen
# ---------------------------------------------------------------------------


class ContainerUpScreen(ModalScreen[str]):
    """Modal dialog to start a container with version selection."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    DEFAULT_CSS = """
    ContainerUpScreen {
        align: center middle;
    }

    ContainerUpScreen > Vertical {
        background: $surface;
        border: round $primary;
        padding: 1 2;
        width: 60;
        max-height: 80%;
        height: auto;
    }

    ContainerUpScreen #title-label {
        text-style: bold;
        color: $primary;
        width: 100%;
        text-align: center;
        margin-bottom: 1;
    }

    ContainerUpScreen #profile-label {
        margin-bottom: 1;
        color: $text-muted;
    }

    ContainerUpScreen #version-label {
        margin-bottom: 0;
        color: $text-muted;
    }

    ContainerUpScreen RadioSet {
        margin-bottom: 1;
        height: auto;
    }

    ContainerUpScreen #custom-tag-input {
        margin-bottom: 1;
        display: none;
    }

    ContainerUpScreen .buttons {
        layout: horizontal;
        height: 3;
        align: center middle;
        margin-top: 1;
    }

    ContainerUpScreen .buttons Button {
        margin: 0 1;
    }

    ContainerUpScreen #loading-area {
        height: auto;
        align: center middle;
        display: none;
    }

    ContainerUpScreen #loading-area LoadingIndicator {
        height: 3;
    }
    """

    def __init__(self, profile_name: str) -> None:
        super().__init__()
        self.profile_name = profile_name

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-dialog"):
            yield Static("Start Container", id="title-label")
            yield Static(f"Profile: [b]{self.profile_name}[/b]", id="profile-label")
            yield Label("Version", id="version-label")
            with RadioSet(id="version-radio"):
                yield RadioButton(VERSION_LOCAL_LATEST, value=True)
                yield RadioButton(VERSION_OFFICIAL)
                yield RadioButton(VERSION_NIGHTLY)
                yield RadioButton(VERSION_DEV_BUILD)
                yield RadioButton(VERSION_CUSTOM_TAG)
            yield Input(
                placeholder="Enter custom image tag...",
                id="custom-tag-input",
            )
            with Vertical(id="loading-area"):
                yield LoadingIndicator()
                yield Static("Starting container...", id="loading-text")
            with Horizontal(classes="buttons"):
                yield Button("Start", variant="primary", id="start-btn")
                yield Button("Cancel", variant="default", id="cancel-btn")

    @on(RadioSet.Changed, "#version-radio")
    def _on_version_changed(self, event: RadioSet.Changed) -> None:
        """Show/hide custom tag input based on radio selection."""
        custom_input = self.query_one("#custom-tag-input", Input)
        if event.pressed.label.plain == VERSION_CUSTOM_TAG:
            custom_input.styles.display = "block"
            custom_input.focus()
        else:
            custom_input.styles.display = "none"

    @on(Button.Pressed, "#cancel-btn")
    def _on_cancel(self) -> None:
        self.dismiss("")

    def action_cancel(self) -> None:
        self.dismiss("")

    @on(Button.Pressed, "#start-btn")
    def _on_start(self) -> None:
        self._do_start()

    @work(exclusive=True)
    async def _do_start(self) -> None:
        """Start the container in a background worker."""
        # Determine version parameters from radio selection
        radio_set = self.query_one("#version-radio", RadioSet)
        pressed_index = radio_set.pressed_index
        labels = [
            VERSION_LOCAL_LATEST,
            VERSION_OFFICIAL,
            VERSION_NIGHTLY,
            VERSION_DEV_BUILD,
            VERSION_CUSTOM_TAG,
        ]
        selected = labels[pressed_index] if 0 <= pressed_index < len(labels) else VERSION_LOCAL_LATEST

        use_dev = False
        tag = ""

        if selected == VERSION_LOCAL_LATEST:
            # No extra args - uses local latest image
            pass
        elif selected == VERSION_OFFICIAL:
            tag = "latest"
        elif selected == VERSION_NIGHTLY:
            tag = "nightly"
        elif selected == VERSION_DEV_BUILD:
            use_dev = True
        elif selected == VERSION_CUSTOM_TAG:
            tag = self.query_one("#custom-tag-input", Input).value.strip()
            if not tag:
                self.app.notify("Please enter a custom tag.", severity="error")
                return

        # Show loading state
        self.query_one("#loading-area").styles.display = "block"
        self.query_one(".buttons").styles.display = "none"

        rc, output = await container_up(self.profile_name, use_dev=use_dev, tag=tag)

        if rc == 0:
            self.dismiss(f"Container '{self.profile_name}' started successfully.")
        else:
            self.query_one("#loading-area").styles.display = "none"
            self.query_one(".buttons").styles.display = "block"
            self.app.notify(
                f"Failed to start container (rc={rc}):\n{output[:200]}",
                severity="error",
                timeout=8,
            )


# ---------------------------------------------------------------------------
# ContainerDownScreen
# ---------------------------------------------------------------------------


class ContainerDownScreen(ModalScreen[str]):
    """Confirmation modal to stop a container."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    DEFAULT_CSS = """
    ContainerDownScreen {
        align: center middle;
    }

    ContainerDownScreen > Vertical {
        background: $surface;
        border: round $error;
        padding: 1 2;
        width: 50;
        height: auto;
    }

    ContainerDownScreen #title-label {
        text-style: bold;
        color: $error;
        width: 100%;
        text-align: center;
        margin-bottom: 1;
    }

    ContainerDownScreen #confirm-text {
        margin-bottom: 1;
        width: 100%;
        text-align: center;
    }

    ContainerDownScreen .buttons {
        layout: horizontal;
        height: 3;
        align: center middle;
        margin-top: 1;
    }

    ContainerDownScreen .buttons Button {
        margin: 0 1;
    }

    ContainerDownScreen #loading-area {
        height: auto;
        align: center middle;
        display: none;
    }

    ContainerDownScreen #loading-area LoadingIndicator {
        height: 3;
    }
    """

    def __init__(self, profile_name: str) -> None:
        super().__init__()
        self.profile_name = profile_name

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-dialog"):
            yield Static("Stop Container", id="title-label")
            yield Static(
                f"Stop container [b]{self.profile_name}[/b]?",
                id="confirm-text",
            )
            with Vertical(id="loading-area"):
                yield LoadingIndicator()
                yield Static("Stopping container...", id="loading-text")
            with Horizontal(classes="buttons"):
                yield Button("Stop", variant="error", id="stop-btn")
                yield Button("Cancel", variant="default", id="cancel-btn")

    @on(Button.Pressed, "#cancel-btn")
    def _on_cancel(self) -> None:
        self.dismiss("")

    def action_cancel(self) -> None:
        self.dismiss("")

    @on(Button.Pressed, "#stop-btn")
    def _on_stop(self) -> None:
        self._do_stop()

    @work(exclusive=True)
    async def _do_stop(self) -> None:
        """Stop the container in a background worker."""
        self.query_one("#loading-area").styles.display = "block"
        self.query_one(".buttons").styles.display = "none"

        rc, output = await container_down(self.profile_name)

        if rc == 0:
            self.dismiss(f"Container '{self.profile_name}' stopped.")
        else:
            self.query_one("#loading-area").styles.display = "none"
            self.query_one(".buttons").styles.display = "block"
            self.app.notify(
                f"Failed to stop container (rc={rc}):\n{output[:200]}",
                severity="error",
                timeout=8,
            )


# ---------------------------------------------------------------------------
# LogScreen
# ---------------------------------------------------------------------------


class LogScreen(Screen):
    """Full-screen log viewer that streams container logs in real-time."""

    DEFAULT_CSS = """
    LogScreen {
        layout: vertical;
    }

    LogScreen #log-header {
        dock: top;
        height: 1;
        color: $text-muted;
        text-style: bold;
        padding: 0 2;
        margin: 1 0;
    }

    LogScreen RichLog {
        height: 1fr;
        margin: 0 1 1 1;
        border: round $primary 40%;
    }
    """

    BINDINGS = [
        Binding("q", "go_back", "Back"),
        Binding("escape", "go_back", "Back"),
    ]

    def __init__(self, container_name: str) -> None:
        super().__init__()
        self.container_name = container_name

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(
            f"Logs: [b]{self.container_name}[/b]  (press [b]q[/b] or [b]Esc[/b] to go back)",
            id="log-header",
        )
        yield RichLog(highlight=True, markup=True, auto_scroll=True, id="log-view")

    def on_mount(self) -> None:
        self._stream_logs()

    @work(exclusive=True)
    async def _stream_logs(self) -> None:
        """Stream container logs into the RichLog widget."""
        log_widget = self.query_one(RichLog)
        try:
            async for line in stream_container_logs(self.container_name):
                log_widget.write(line)
        except Exception as exc:
            log_widget.write(f"\n[red]Log stream error: {exc}[/red]")

    def action_go_back(self) -> None:
        self.app.pop_screen()
