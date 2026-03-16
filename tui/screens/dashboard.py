"""Dashboard screen - main screen showing all profiles and their statuses."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    OptionList,
    Static,
)
from textual.widgets.option_list import Option
from textual import work, on

from tui.backend import (
    ContainerStatus,
    get_container_statuses,
    container_down,
    load_profile,
)


class DashboardScreen(Screen):
    """Main dashboard displaying all vLLM profiles and their statuses."""

    BINDINGS = [
        # Primary: visible in footer
        Binding("w", "quick_setup", "Quick Setup"),
        Binding("n", "new_profile", "New"),
        Binding("s", "system_info", "System"),
        Binding("c", "configs", "Configs"),
        # Direct shortcuts: power-user (hidden, but listed in ? help)
        Binding("u", "start_container", show=False),
        Binding("d", "stop_container", show=False),
        Binding("l", "view_logs", show=False),
        Binding("e", "edit_profile", show=False),
        Binding("x", "delete_profile", show=False),
        Binding("r", "refresh", show=False),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._statuses: list[ContainerStatus] = []
        self._refresh_timer = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="dashboard-header"):
            yield Static("vLLM Compose", id="dashboard-brand")
            yield Static("Docker Compose based Multi-Model Serving", id="dashboard-subtitle")
        yield Static("", id="status-bar")
        yield DataTable(id="profile-table", cursor_type="row")
        yield Static(
            "\n\n\n\n  No profiles yet\n\n"
            "  [dim][b]w[/b] Quick Setup  ·  [b]n[/b] New Profile[/dim]",
            id="empty-state",
        )
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#profile-table", DataTable)
        table.add_columns("Name", "Status", "GPU", "Port", "Model", "LoRA")
        self._load_statuses()
        self._refresh_timer = self.set_interval(5, self._load_statuses)

    def on_screen_suspend(self) -> None:
        if self._refresh_timer is not None:
            self._refresh_timer.pause()

    def on_screen_resume(self) -> None:
        self._load_statuses()
        if self._refresh_timer is not None:
            self._refresh_timer.resume()

    @work(exclusive=True, group="refresh")
    async def _load_statuses(self) -> None:
        statuses = await get_container_statuses()
        self._statuses = statuses
        self._update_table(statuses)

    def _update_table(self, statuses: list[ContainerStatus]) -> None:
        table = self.query_one("#profile-table", DataTable)
        empty_state = self.query_one("#empty-state")
        status_bar = self.query_one("#status-bar", Static)

        # Preserve cursor position
        selected_key: str | None = None
        if table.row_count > 0:
            try:
                row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
                selected_key = str(row_key)
            except (KeyError, IndexError):
                pass

        table.clear()

        if not statuses:
            table.styles.display = "none"
            empty_state.styles.display = "block"
            status_bar.update("")
            return

        table.styles.display = "block"
        empty_state.styles.display = "none"

        running_count = sum(1 for s in statuses if s.running)
        status_bar.update(
            f" {len(statuses)} profiles  ·  {running_count} running"
            "  ·  [dim]Enter = actions[/dim]"
        )

        for s in statuses:
            if s.running:
                status_cell = "[green]● running[/]"
            else:
                status_cell = "[dim]○ stopped[/]"
            lora_cell = "✓" if s.lora else ""
            model_short = s.model.split("/")[-1] if s.model else ""
            table.add_row(
                s.profile_name,
                status_cell,
                s.gpu_id,
                s.port,
                model_short,
                lora_cell,
                key=s.profile_name,
            )

        # Restore cursor position
        if selected_key:
            for idx, s in enumerate(statuses):
                if s.profile_name == selected_key:
                    table.move_cursor(row=idx)
                    break

    def _get_selected_profile(self) -> str | None:
        table = self.query_one("#profile-table", DataTable)
        if table.row_count == 0:
            return None
        try:
            row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
            return str(row_key)
        except (KeyError, IndexError):
            return None

    # ----- Enter (DataTable row selected): Action Menu -----

    @on(DataTable.RowSelected, "#profile-table")
    def _on_row_selected(self, event: DataTable.RowSelected) -> None:
        profile = str(event.row_key.value)
        running = any(
            s.running for s in self._statuses if s.profile_name == profile
        )
        captured_profile = profile

        def on_action(action: str) -> None:
            if action:
                self._handle_action(action, captured_profile)

        self.app.push_screen(
            ProfileActionScreen(profile, running),
            callback=on_action,
        )

    def _handle_action(self, action: str, profile_name: str) -> None:
        if action == "start":
            from tui.screens.container import ContainerUpScreen
            self.app.push_screen(ContainerUpScreen(profile_name), callback=self._on_profile_saved)
        elif action == "stop":
            self._confirm_stop(profile_name)
        elif action == "logs":
            from tui.screens.container import LogScreen
            p = load_profile(profile_name)
            self.app.push_screen(LogScreen(p.container_name))
        elif action == "edit_profile":
            from tui.screens.profile import ProfileFormScreen
            p = load_profile(profile_name)
            self.app.push_screen(ProfileFormScreen(p), callback=self._on_profile_saved)
        elif action == "edit_config":
            p = load_profile(profile_name)
            if p.config_name:
                from tui.screens.config import ConfigFormScreen
                self.app.push_screen(
                    ConfigFormScreen(config_name=p.config_name),
                    callback=self._on_profile_saved,
                )
            else:
                self.notify("No config linked to this profile.", severity="warning")
        elif action == "delete":
            is_running = any(
                s.running for s in self._statuses if s.profile_name == profile_name
            )
            if is_running:
                self.notify("Cannot delete: container is running. Stop it first.", severity="error")
                return
            from tui.screens.profile import ProfileDeleteScreen
            self.app.push_screen(
                ProfileDeleteScreen(profile_name), callback=self._on_profile_saved
            )

    # ----- Direct shortcut actions -----

    def action_refresh(self) -> None:
        self._load_statuses()

    def action_start_container(self) -> None:
        profile = self._get_selected_profile()
        if profile is None:
            self.notify("No profile selected.", severity="warning")
            return
        from tui.screens.container import ContainerUpScreen
        self.app.push_screen(ContainerUpScreen(profile), callback=self._on_profile_saved)

    def action_stop_container(self) -> None:
        profile = self._get_selected_profile()
        if profile is None:
            self.notify("No profile selected.", severity="warning")
            return
        self._confirm_stop(profile)

    def _confirm_stop(self, profile_name: str) -> None:
        def on_confirm(result: bool) -> None:
            if result:
                self._do_stop(profile_name)

        self.app.push_screen(
            ConfirmStopScreen(profile_name), callback=on_confirm
        )

    @work(exclusive=False)
    async def _do_stop(self, profile_name: str) -> None:
        self.notify(f"Stopping {profile_name}...")
        rc, output = await container_down(profile_name)
        if rc == 0:
            self.notify(f"Stopped {profile_name}.", severity="information")
        else:
            self.notify(f"Error stopping {profile_name}: {output}", severity="error")
        self._load_statuses()

    def action_view_logs(self) -> None:
        profile = self._get_selected_profile()
        if profile is None:
            self.notify("No profile selected.", severity="warning")
            return
        from tui.screens.container import LogScreen
        p = load_profile(profile)
        self.app.push_screen(LogScreen(p.container_name))

    def action_new_profile(self) -> None:
        from tui.screens.profile import ProfileFormScreen
        self.app.push_screen(ProfileFormScreen(), callback=self._on_profile_saved)

    def action_edit_profile(self) -> None:
        profile = self._get_selected_profile()
        if profile is None:
            self.notify("No profile selected.", severity="warning")
            return
        from tui.screens.profile import ProfileFormScreen
        p = load_profile(profile)
        self.app.push_screen(ProfileFormScreen(p), callback=self._on_profile_saved)

    def _on_profile_saved(self, result: object = None) -> None:
        self._load_statuses()

    def action_delete_profile(self) -> None:
        profile = self._get_selected_profile()
        if profile is None:
            self.notify("No profile selected.", severity="warning")
            return
        is_running = any(
            s.running for s in self._statuses if s.profile_name == profile
        )
        if is_running:
            self.notify("Cannot delete: container is running. Stop it first.", severity="error")
            return
        from tui.screens.profile import ProfileDeleteScreen
        self.app.push_screen(
            ProfileDeleteScreen(profile), callback=self._on_profile_saved
        )

    def action_system_info(self) -> None:
        self.app.switch_screen("system")

    def action_configs(self) -> None:
        self.app.switch_screen("configs")

    def action_quick_setup(self) -> None:
        from tui.screens.quick_setup import QuickSetupScreen
        self.app.push_screen(QuickSetupScreen(), callback=self._on_profile_saved)


# ---------------------------------------------------------------------------
# Profile Action Menu - the key UX: Enter on a profile opens this
# ---------------------------------------------------------------------------


class ProfileActionScreen(ModalScreen[str]):
    """Context action menu for a selected profile.

    Shows relevant actions based on container state (running/stopped).
    Returns the action id string on selection, empty string on cancel.
    """

    BINDINGS = [Binding("escape", "cancel", show=False)]

    DEFAULT_CSS = """
    ProfileActionScreen {
        align: center middle;
    }
    ProfileActionScreen > Vertical {
        background: $surface;
        border: round $primary;
        padding: 1 2;
        width: 42;
        height: auto;
    }
    ProfileActionScreen #action-title {
        text-style: bold;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }
    ProfileActionScreen OptionList {
        height: auto;
        max-height: 14;
    }
    """

    def __init__(self, profile_name: str, running: bool) -> None:
        super().__init__()
        self.profile_name = profile_name
        self._profile_running = running

    def compose(self) -> ComposeResult:
        if self._profile_running:
            status = "[green]● running[/]"
        else:
            status = "[dim]○ stopped[/]"

        options: list[Option] = []
        if self._profile_running:
            options.append(Option("■ Stop Container", id="stop"))
            options.append(Option("◉ View Logs", id="logs"))
        else:
            options.append(Option("▶ Start Container", id="start"))
        options.append(Option("✎ Edit Profile", id="edit_profile"))
        options.append(Option("⚙ Edit Config", id="edit_config"))
        options.append(Option("✗ Delete Profile", id="delete"))

        with Vertical():
            yield Static(
                f"{self.profile_name}  {status}", id="action-title"
            )
            yield OptionList(*options, id="action-list")

    @on(OptionList.OptionSelected, "#action-list")
    def _on_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id or "")

    def action_cancel(self) -> None:
        self.dismiss("")


# ---------------------------------------------------------------------------
# Stop confirmation
# ---------------------------------------------------------------------------


class ConfirmStopScreen(ModalScreen[bool]):
    """Modal dialog to confirm stopping a container."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    DEFAULT_CSS = """
    ConfirmStopScreen {
        align: center middle;
    }
    ConfirmStopScreen > Vertical {
        background: $surface;
        border: round $error;
        padding: 1 2;
        width: 50;
        height: auto;
    }
    ConfirmStopScreen #confirm-message {
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }
    """

    def __init__(self, profile_name: str) -> None:
        super().__init__()
        self._profile_name = profile_name

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(
                f"Stop container [b]{self._profile_name}[/b]?",
                id="confirm-message",
            ),
            Horizontal(
                Button("Yes, stop", id="confirm-yes", variant="error"),
                Button("Cancel", id="confirm-no", variant="default"),
                classes="form-buttons",
            ),
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-yes")

    def action_cancel(self) -> None:
        self.dismiss(False)
