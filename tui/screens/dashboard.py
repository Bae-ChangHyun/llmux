"""Dashboard screen - main screen showing all profiles and their statuses."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, DataTable, Footer, Header, Static
from textual import work

from tui.backend import (
    ContainerStatus,
    get_container_statuses,
    container_down,
    load_profile,
)


class DashboardScreen(Screen):
    """Main dashboard displaying all vLLM profiles and their statuses."""

    BINDINGS = [
        Binding("u", "start_container", "Start", show=True),
        Binding("d", "stop_container", "Stop", show=True),
        Binding("l", "view_logs", "Logs", show=True),
        Binding("n", "new_profile", "New Profile", show=True),
        Binding("e", "edit_profile", "Edit Profile", show=True),
        Binding("x", "delete_profile", "Delete", show=True),
        Binding("s", "system_info", "System Info", show=True),
        Binding("c", "configs", "Configs", show=True),
        Binding("w", "quick_setup", "Quick Setup", show=True),
        Binding("r", "refresh", "Refresh", show=True),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._statuses: list[ContainerStatus] = []
        self._refresh_timer = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("vLLM Container Dashboard", id="dashboard-title")
        yield DataTable(id="profile-table", cursor_type="row")
        yield Horizontal(
            Button("Start (u)", id="btn-start", variant="success"),
            Button("Stop (d)", id="btn-stop", variant="error"),
            Button("Logs (l)", id="btn-logs", variant="primary"),
            Button("New Profile (n)", id="btn-new", variant="default"),
            Button("System Info (s)", id="btn-system", variant="default"),
            id="action-bar",
        )
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#profile-table", DataTable)
        table.add_columns("Name", "Status", "GPU", "Port", "Model", "LoRA")
        self._load_statuses()
        self._refresh_timer = self.set_interval(5, self._load_statuses)

    @work(exclusive=True)
    async def _load_statuses(self) -> None:
        """Fetch container statuses and update the table."""
        statuses = await get_container_statuses()
        self._statuses = statuses
        self._update_table(statuses)

    def _update_table(self, statuses: list[ContainerStatus]) -> None:
        """Rebuild the DataTable rows from the given statuses."""
        table = self.query_one("#profile-table", DataTable)
        table.clear()
        for s in statuses:
            if s.running:
                status_cell = "[green]\u25cf running[/]"
            else:
                status_cell = "[dim]\u25cb stopped[/]"
            lora_cell = "\u2713" if s.lora else ""
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

    def _get_selected_profile(self) -> str | None:
        """Return the profile name of the currently selected row, or None."""
        table = self.query_one("#profile-table", DataTable)
        if table.row_count == 0:
            return None
        try:
            row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
            return str(row_key)
        except Exception:
            return None

    # ----- Actions -----

    def action_refresh(self) -> None:
        self._load_statuses()

    def action_start_container(self) -> None:
        profile = self._get_selected_profile()
        if profile is None:
            self.notify("No profile selected.", severity="warning")
            return
        from tui.screens.container import ContainerUpScreen  # lazy import

        self.app.push_screen(ContainerUpScreen(profile))

    def action_stop_container(self) -> None:
        profile = self._get_selected_profile()
        if profile is None:
            self.notify("No profile selected.", severity="warning")
            return
        self._confirm_stop(profile)

    def _confirm_stop(self, profile_name: str) -> None:
        """Ask for confirmation before stopping a container."""

        def on_confirm(result: bool) -> None:
            if result:
                self._do_stop(profile_name)

        self.app.push_screen(
            ConfirmStopScreen(profile_name),
            callback=on_confirm,
        )

    @work(exclusive=True, group="stop")
    async def _do_stop(self, profile_name: str) -> None:
        """Stop the container for the given profile."""
        self.notify(f"Stopping {profile_name}...")
        rc, output = await container_down(profile_name)
        if rc == 0:
            self.notify(f"Stopped {profile_name}.", severity="information")
        else:
            self.notify(f"Error stopping {profile_name}: {output}", severity="error")
        # Trigger a refresh (calls the @work-decorated method which runs in background)
        self._load_statuses()

    def action_view_logs(self) -> None:
        profile = self._get_selected_profile()
        if profile is None:
            self.notify("No profile selected.", severity="warning")
            return
        from tui.screens.container import LogScreen  # lazy import

        p = load_profile(profile)
        self.app.push_screen(LogScreen(p.container_name))

    def action_new_profile(self) -> None:
        from tui.screens.profile import ProfileFormScreen  # lazy import

        self.app.push_screen(ProfileFormScreen(), callback=self._on_profile_saved)

    def action_edit_profile(self) -> None:
        profile = self._get_selected_profile()
        if profile is None:
            self.notify("No profile selected.", severity="warning")
            return
        from tui.screens.profile import ProfileFormScreen  # lazy import

        p = load_profile(profile)
        self.app.push_screen(ProfileFormScreen(p), callback=self._on_profile_saved)

    def _on_profile_saved(self, result: object = None) -> None:
        """Callback after a profile form is dismissed."""
        self._load_statuses()

    def action_delete_profile(self) -> None:
        profile = self._get_selected_profile()
        if profile is None:
            self.notify("No profile selected.", severity="warning")
            return
        from tui.screens.profile import ProfileDeleteScreen  # lazy import

        self.app.push_screen(ProfileDeleteScreen(profile), callback=self._on_profile_saved)

    def action_system_info(self) -> None:
        from tui.screens.system import SystemScreen  # lazy import

        self.app.push_screen(SystemScreen())

    def action_configs(self) -> None:
        from tui.screens.config import ConfigListScreen  # lazy import

        self.app.push_screen(ConfigListScreen())

    def action_quick_setup(self) -> None:
        from tui.screens.quick_setup import QuickSetupScreen  # lazy import

        self.app.push_screen(QuickSetupScreen(), callback=self._on_profile_saved)

    # ----- Button handlers -----

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_action_map = {
            "btn-start": "start_container",
            "btn-stop": "stop_container",
            "btn-logs": "view_logs",
            "btn-new": "new_profile",
            "btn-system": "system_info",
        }
        action = button_action_map.get(event.button.id, "")
        if action:
            self.run_action(action)


# ---------------------------------------------------------------------------
# Small modal screen for stop confirmation
# ---------------------------------------------------------------------------


class ConfirmStopScreen(ModalScreen[bool]):
    """Modal dialog to confirm stopping a container."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, profile_name: str) -> None:
        super().__init__()
        self._profile_name = profile_name

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(
                f"Stop container for [b]{self._profile_name}[/b]?",
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
