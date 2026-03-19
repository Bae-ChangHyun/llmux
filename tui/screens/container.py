"""Container management screens: up, down, and log viewer."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
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
    check_port_conflict,
    load_profile,
    stream_container_up,
    stream_container_logs,
    get_local_latest_tag,
    get_dockerhub_release_version,
    get_dockerhub_nightly_date,
)


# ---------------------------------------------------------------------------
# Version option IDs (stable keys for logic, labels updated dynamically)
# ---------------------------------------------------------------------------

VER_LOCAL = "local_latest"
VER_OFFICIAL = "official"
VER_NIGHTLY = "nightly"
VER_DEV = "dev_build"
VER_CUSTOM = "custom_tag"


# ---------------------------------------------------------------------------
# ContainerUpScreen
# ---------------------------------------------------------------------------


class ContainerUpScreen(ModalScreen[str]):
    """Modal dialog to start a container with version selection."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("q", "cancel", "Quit", show=False),
    ]

    DEFAULT_CSS = """
    ContainerUpScreen {
        align: center middle;
    }

    ContainerUpScreen > Vertical {
        background: $surface;
        border: round $primary;
        padding: 1 2;
        width: 65;
        max-height: 90%;
    }

    ContainerUpScreen #title-label {
        text-style: bold;
        color: $primary;
        width: 100%;
        text-align: center;
    }

    ContainerUpScreen #profile-label {
        color: $text-muted;
    }

    ContainerUpScreen #version-label {
        margin-bottom: 0;
        color: $text-muted;
    }

    ContainerUpScreen RadioSet {
        height: auto;
    }

    ContainerUpScreen #custom-tag-input {
        margin-bottom: 1;
        display: none;
    }

    ContainerUpScreen .buttons {
        height: 1;
        align: center middle;
        margin-top: 1;
    }

    ContainerUpScreen #startup-area {
        display: none;
        height: 1fr;
    }

    ContainerUpScreen #startup-status {
        height: 1;
        margin-bottom: 0;
    }

    ContainerUpScreen #startup-log {
        height: 1fr;
        border: round $primary 40%;
    }
    """

    def __init__(self, profile_name: str) -> None:
        super().__init__()
        self.profile_name = profile_name
        self._profile = load_profile(profile_name)

    def compose(self) -> ComposeResult:
        # UX: Guard - config must be set before starting
        if not self._profile.config_name:
            with Vertical(id="modal-dialog"):
                yield Static("Start Container", id="title-label")
                yield Static(
                    f"Profile [b]{self.profile_name}[/b] has no config linked.\n"
                    "Please edit the profile and select a config first.",
                    id="profile-label",
                )
                with Horizontal(classes="buttons"):
                    yield Button("Close", variant="default", id="cancel-btn")
            return
        with Vertical(id="modal-dialog"):
            yield Static("Start Container", id="title-label")
            yield Static(f"Profile: [b]{self.profile_name}[/b]", id="profile-label")
            with VerticalScroll(id="version-scroll"):
                yield Label("Version", id="version-label")
                with RadioSet(id="version-radio"):
                    yield RadioButton("Local Latest  (loading...)", id=VER_LOCAL, value=True)
                    yield RadioButton("Official Release  (loading...)", id=VER_OFFICIAL)
                    yield RadioButton("Nightly  (loading...)", id=VER_NIGHTLY)
                    yield RadioButton("Dev Build  (vllm-dev)", id=VER_DEV)
                    yield RadioButton("Custom Tag", id=VER_CUSTOM)
                yield Input(
                    placeholder="Enter custom image tag...",
                    id="custom-tag-input",
                )
            with Vertical(id="startup-area"):
                yield Static("", id="startup-status")
                yield RichLog(highlight=True, auto_scroll=True, id="startup-log")
            with Horizontal(classes="buttons"):
                yield Button("Start", variant="primary", id="start-btn")
                yield Button("Cancel", variant="default", id="cancel-btn")

    def on_mount(self) -> None:
        if self._profile.config_name:
            self._fetch_version_info()

    @work(exclusive=False)
    async def _fetch_version_info(self) -> None:
        """Fetch version info and update radio button labels."""
        try:
            radio_set = self.query_one("#version-radio", RadioSet)
        except Exception:
            return

        # Local latest
        local_tag = await get_local_latest_tag()
        try:
            btn = radio_set.query_one(f"#{VER_LOCAL}", RadioButton)
            if local_tag == "none":
                btn.label = "Local Latest  (no images)"
            else:
                btn.label = f"Local Latest  ({local_tag})"
        except Exception:
            pass

        # Official release
        release_ver = await get_dockerhub_release_version()
        try:
            btn = radio_set.query_one(f"#{VER_OFFICIAL}", RadioButton)
            btn.label = f"Official Release  ({release_ver})"
        except Exception:
            pass

        # Nightly
        nightly_date = await get_dockerhub_nightly_date()
        try:
            btn = radio_set.query_one(f"#{VER_NIGHTLY}", RadioButton)
            btn.label = f"Nightly  ({nightly_date})"
        except Exception:
            pass

    @on(RadioSet.Changed, "#version-radio")
    def _on_version_changed(self, event: RadioSet.Changed) -> None:
        """Show/hide custom tag input based on radio selection."""
        custom_input = self.query_one("#custom-tag-input", Input)
        pressed = event.pressed
        if pressed and pressed.id == VER_CUSTOM:
            custom_input.styles.display = "block"
            custom_input.focus()
        else:
            custom_input.styles.display = "none"

    @on(Button.Pressed, "#cancel-btn")
    def _on_cancel(self) -> None:
        self.workers.cancel_all()
        self.dismiss("")

    def action_cancel(self) -> None:
        self.workers.cancel_all()
        self.dismiss("")

    @on(Button.Pressed, "#start-btn")
    def _on_start(self) -> None:
        self._do_start()

    @work(exclusive=True)
    async def _do_start(self) -> None:
        """Start the container in a background worker."""
        # Determine version from radio selection
        radio_set = self.query_one("#version-radio", RadioSet)
        pressed = radio_set.pressed_button
        selected_id = pressed.id if pressed else VER_LOCAL

        use_dev = False
        tag = ""

        if selected_id == VER_LOCAL:
            pass
        elif selected_id == VER_OFFICIAL:
            tag = "latest"
        elif selected_id == VER_NIGHTLY:
            tag = "nightly"
        elif selected_id == VER_DEV:
            use_dev = True
        elif selected_id == VER_CUSTOM:
            tag = self.query_one("#custom-tag-input", Input).value.strip()
            if not tag:
                self.app.notify("Please enter a custom tag.", severity="error")
                return

        # Check port conflict before starting
        conflict = await check_port_conflict(self._profile)
        if conflict:
            self.app.notify(
                f"Port {self._profile.port} is already used by profile '{conflict}'.",
                severity="error",
                timeout=5,
            )
            return

        # Switch to startup log view
        try:
            self.query_one("#startup-area").styles.display = "block"
            self.query_one("#version-scroll").styles.display = "none"
            self.query_one(".buttons").styles.display = "none"
            status = self.query_one("#startup-status", Static)
            log_widget = self.query_one("#startup-log", RichLog)
            status.update("[bold]Starting container...[/bold]")
        except Exception:  # Screen may already be dismissed
            return

        # Stream run.sh output in real-time
        rc = -1
        async for msg_type, data in stream_container_up(self.profile_name, use_dev=use_dev, tag=tag):
            if msg_type == "log":
                try:
                    log_widget.write(data)
                except Exception:
                    pass
            elif msg_type == "rc":
                rc = data

        try:
            # Show close button for both success and failure
            self.query_one("#start-btn").styles.display = "none"
            self.query_one(".buttons").styles.display = "block"
            if rc == 0:
                status.update("[green bold]Container started. Logs: (Esc/q to close)[/green bold]")
                try:
                    async for line in stream_container_logs(self._profile.container_name):
                        log_widget.write(line)
                except Exception:
                    pass
            else:
                status.update(f"[red bold]Failed to start (rc={rc})[/red bold]")
        except Exception:  # Screen may already be dismissed
            pass


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
        self.workers.cancel_all()
        self.app.pop_screen()
