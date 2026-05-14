"""Container start screen for llama.cpp — mirrors vllm's ContainerUpScreen.

Three startup modes (vs vllm's five — llama.cpp has a single official
ghcr.io tag so "Local Latest / Official / Nightly" collapses into one):

  - Default Image      ghcr.io/ggml-org/llama.cpp:server-cuda (LLAMACPP_IMAGE)
  - Dev Build          llamacpp-dev:<branch> — builds on demand from
                       LLAMACPP_REPO_URL @ LLAMACPP_BRANCH (.env.common
                       defaults, editable here)
  - Custom Tag         arbitrary <repo>:<tag> the user types in

`Start` then drives backend_runtime.stream_container_up() with the
matching keyword args — same shape as the vllm screen.
"""

from __future__ import annotations

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Input,
    Label,
    RadioButton,
    RadioSet,
    RichLog,
    Static,
)

from tui.backends.llamacpp import backend
from tui.backends.llamacpp.backend_runtime import (
    LLAMACPP_DEV_SPEC,
    get_dev_build_defaults,
    stream_container_up,
)
from tui.common.dev_build import list_local_dev_images


VER_DEFAULT = "default_image"
VER_DEV = "dev_build"
VER_CUSTOM = "custom_tag"


class ContainerUpScreen(Screen):
    """Full-screen llama.cpp startup picker + log viewer."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("q", "cancel", "Quit", show=False),
        Binding("f", "toggle_follow", "Follow on/off"),
    ]

    DEFAULT_CSS = """
    ContainerUpScreen {
        layout: vertical;
    }

    ContainerUpScreen > Vertical {
        width: 100%;
        height: 100%;
        padding: 0 1;
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

    ContainerUpScreen #version-help {
        color: $text-muted;
        margin-top: 0;
        margin-bottom: 0;
    }

    ContainerUpScreen RadioSet {
        height: auto;
    }

    ContainerUpScreen RadioSet > RadioButton.-on > .toggle--button {
        color: #34d399;
        background: $surface;
        text-style: bold;
    }

    ContainerUpScreen RadioSet > RadioButton.-on > .toggle--label {
        color: #34d399;
        text-style: bold;
    }

    ContainerUpScreen #custom-tag-input {
        margin-bottom: 1;
        display: none;
    }

    ContainerUpScreen #dev-build-options {
        display: none;
        height: auto;
        margin-bottom: 1;
    }

    ContainerUpScreen .dev-build-row {
        height: auto;
        margin-bottom: 1;
    }

    ContainerUpScreen .dev-build-row Label {
        width: 10;
        color: $text-muted;
        padding-top: 1;
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
        margin: 0;
    }
    """

    def __init__(self, profile_name: str) -> None:
        super().__init__()
        self.profile_name = profile_name
        self._profile = backend.load_profile(profile_name)
        self._dev_repo_url, self._dev_branch = get_dev_build_defaults()
        self._has_local_dev: bool = False

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-dialog"):
            yield Static("Start llama.cpp Container", id="title-label")
            yield Static(f"Profile: [b]{self.profile_name}[/b]", id="profile-label")
            if not self._profile.config_name:
                yield Static(
                    "[yellow]No config linked. A default config will be generated on start.[/yellow]"
                )

            # If the profile is already pinned to a dev image, surface that.
            if self._profile.image_tag.startswith(f"{LLAMACPP_DEV_SPEC.image_prefix}:"):
                yield Static(
                    f"[cyan]Profile pinned to: {self._profile.image_tag}[/cyan]  "
                    "[dim](selecting Default Image or Custom Tag will override for this start only)[/dim]",
                )

            with Vertical(id="version-scroll"):
                yield Label("Version", id="version-label")
                yield Static(
                    "[dim]Click or use ↑↓ + Enter/Space to select[/dim]",
                    id="version-help",
                )
                with RadioSet(id="version-radio"):
                    yield RadioButton(
                        "Default Image  (ghcr.io/ggml-org/llama.cpp:server-cuda)",
                        id=VER_DEFAULT,
                        value=True,
                    )
                    yield RadioButton(
                        "Dev Build  (llamacpp-dev:<branch>)  (loading...)",
                        id=VER_DEV,
                    )
                    yield RadioButton("Custom Tag", id=VER_CUSTOM)
                yield Input(
                    placeholder="Enter custom image tag (e.g. llamacpp-dev:mtp-clean)...",
                    id="custom-tag-input",
                )
                with Vertical(id="dev-build-options"):
                    with Horizontal(classes="dev-build-row"):
                        yield Label("Repo URL")
                        yield Input(
                            value=self._dev_repo_url,
                            placeholder="https://github.com/ggml-org/llama.cpp.git",
                            id="dev-repo-input",
                        )
                    with Horizontal(classes="dev-build-row"):
                        yield Label("Branch")
                        yield Input(
                            value=self._dev_branch,
                            placeholder="master",
                            id="dev-branch-input",
                        )
            with Vertical(id="startup-area"):
                yield Static("", id="startup-status")
                yield RichLog(
                    highlight=False,
                    markup=False,
                    wrap=False,
                    auto_scroll=True,
                    max_lines=5000,
                    id="startup-log",
                )
            with Horizontal(classes="buttons"):
                yield Button("Start", variant="primary", id="start-btn")
                yield Button("Cancel", variant="default", id="cancel-btn")

    def on_mount(self) -> None:
        self._refresh_dev_label()
        try:
            self.query_one("#version-radio", RadioSet).focus()
        except Exception:
            pass

    @work(exclusive=False)
    async def _refresh_dev_label(self) -> None:
        """Show how many local llamacpp-dev tags exist so the user knows
        whether picking Dev Build will rebuild or reuse."""
        try:
            radio_set = self.query_one("#version-radio", RadioSet)
            btn = radio_set.query_one(f"#{VER_DEV}", RadioButton)
        except Exception:
            return
        images = await list_local_dev_images(LLAMACPP_DEV_SPEC)
        self._has_local_dev = bool(images)
        label = (
            f"Dev Build  ({len(images)} local tag{'s' if len(images) != 1 else ''})"
            if images
            else "Dev Build  (no local tags — will build from source)"
        )
        btn.label = label

    @on(RadioSet.Changed, "#version-radio")
    def _on_version_changed(self, event: RadioSet.Changed) -> None:
        custom_input = self.query_one("#custom-tag-input", Input)
        dev_options = self.query_one("#dev-build-options", Vertical)
        pressed = event.pressed
        if pressed and pressed.id == VER_CUSTOM:
            custom_input.styles.display = "block"
            dev_options.styles.display = "none"
            custom_input.focus()
        elif pressed and pressed.id == VER_DEV:
            custom_input.styles.display = "none"
            dev_options.styles.display = "block"
            self.query_one("#dev-repo-input", Input).focus()
        else:
            custom_input.styles.display = "none"
            dev_options.styles.display = "none"

    def _cleanup(self) -> None:
        self.workers.cancel_all()

    @on(Button.Pressed, "#cancel-btn")
    def _on_cancel(self) -> None:
        self._cleanup()
        self.app.pop_screen()

    def action_cancel(self) -> None:
        self._cleanup()
        self.app.pop_screen()

    @on(Button.Pressed, "#start-btn")
    def _on_start(self) -> None:
        self._do_start()

    @work(exclusive=True, group="llamacpp-start")
    async def _do_start(self) -> None:
        radio_set = self.query_one("#version-radio", RadioSet)
        pressed = radio_set.pressed_button
        selected_id = pressed.id if pressed else VER_DEFAULT

        use_dev = False
        tag = ""
        repo_url = ""
        branch = ""

        if selected_id == VER_DEV:
            use_dev = True
            repo_url = self.query_one("#dev-repo-input", Input).value.strip()
            branch = self.query_one("#dev-branch-input", Input).value.strip()
            if not repo_url:
                self.app.notify("Please enter a repository URL.", severity="error")
                return
            if not branch:
                self.app.notify("Please enter a branch.", severity="error")
                return
        elif selected_id == VER_CUSTOM:
            tag = self.query_one("#custom-tag-input", Input).value.strip()
            if not tag:
                self.app.notify("Please enter a custom tag.", severity="error")
                return

        # VER_DEFAULT falls through with all defaults — backend resolves
        # to the compose default image.

        # Switch to log view
        try:
            self.query_one("#startup-area").styles.display = "block"
            self.query_one("#version-scroll").styles.display = "none"
            self.query_one("#start-btn").styles.display = "none"
            status = self.query_one("#startup-status", Static)
            log_widget = self.query_one("#startup-log", RichLog)
            status.update("[bold]Starting container...[/bold]")
        except Exception:
            return

        rc = -1
        async for msg_type, data in stream_container_up(
            self.profile_name,
            use_dev=use_dev,
            tag=tag,
            repo_url=repo_url,
            branch=branch,
        ):
            if msg_type == "log":
                try:
                    log_widget.write(backend.strip_ansi(data))
                except Exception:
                    pass
            elif msg_type == "rc":
                rc = int(data)

        try:
            if rc == 0:
                status.update(
                    "[green bold]Container started. Logs: (Esc/q to close)[/green bold]"
                )
                try:
                    async for line in backend.stream_logs(
                        self._profile.container_name
                    ):
                        log_widget.write(backend.strip_ansi(line))
                except Exception as exc:
                    log_widget.write(f"Log stream error: {exc}")
            else:
                status.update(f"[red bold]Failed to start (rc={rc})[/red bold]")
        except Exception:
            pass

    def action_toggle_follow(self) -> None:
        try:
            log_widget = self.query_one("#startup-log", RichLog)
        except Exception:
            return
        log_widget.auto_scroll = not log_widget.auto_scroll
        if log_widget.auto_scroll:
            log_widget.scroll_end(animate=False)
        self.notify(
            f"auto-follow: {'ON' if log_widget.auto_scroll else 'OFF'}", timeout=2
        )
