"""Config management screens - form for create/edit and list screen."""

from __future__ import annotations

import re

from textual.app import ComposeResult
from textual.screen import Screen, ModalScreen
from textual.widgets import Button, Static, Label, Input, DataTable, Footer, Header
from textual.containers import Horizontal, Vertical
from textual.binding import Binding
from textual import on

from tui.backend import (
    Config,
    load_config,
    save_config,
    delete_config,
    list_config_names,
)


def _validate_name(name: str) -> bool:
    """Check that name contains only alphanumeric, dash, and underscore."""
    return bool(re.match(r"^[a-zA-Z0-9_-]+$", name))


# ---------------------------------------------------------------------------
# ConfigFormScreen
# ---------------------------------------------------------------------------


class ConfigFormScreen(ModalScreen[str | None]):
    """Modal form for creating or editing a config."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    DEFAULT_CSS = """
    ConfigFormScreen {
        align: center middle;
    }
    ConfigFormScreen > Vertical {
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        width: 65;
        max-height: 85%;
    }
    ConfigFormScreen .form-row {
        height: auto;
        margin-bottom: 1;
    }
    ConfigFormScreen .form-row Label {
        width: 24;
        padding: 1 1 0 0;
    }
    ConfigFormScreen .form-buttons {
        height: auto;
        margin-top: 1;
        align: center middle;
    }
    ConfigFormScreen .form-buttons Button {
        margin: 0 1;
    }
    """

    def __init__(self, config_name: str = "") -> None:
        super().__init__()
        self._config_name = config_name
        self._edit_mode = bool(config_name)

    def compose(self) -> ComposeResult:
        cfg: Config | None = None
        if self._edit_mode:
            cfg = load_config(self._config_name)

        title = f"Edit Config: {self._config_name}" if self._edit_mode else "New Config"

        with Vertical():
            yield Static(f"[b]{title}[/b]", id="form-title")

            with Horizontal(classes="form-row"):
                yield Label("Config Name")
                yield Input(
                    value=cfg.name if cfg else "",
                    placeholder="my-config",
                    id="name-input",
                    disabled=self._edit_mode,
                )

            with Horizontal(classes="form-row"):
                yield Label("Model")
                yield Input(
                    value=cfg.model if cfg else "",
                    placeholder="org/model-name",
                    id="model-input",
                )

            with Horizontal(classes="form-row"):
                yield Label("GPU Memory Utilization")
                yield Input(
                    value=cfg.gpu_memory_utilization if cfg else "",
                    placeholder="0.9",
                    id="gpu-mem-input",
                )

            with Horizontal(classes="form-buttons"):
                yield Button("Save", id="save-btn", variant="primary")
                yield Button("Cancel", id="cancel-btn", variant="default")

    @on(Button.Pressed, "#save-btn")
    def _on_save(self, event: Button.Pressed) -> None:
        name = self.query_one("#name-input", Input).value.strip()
        model = self.query_one("#model-input", Input).value.strip()
        gpu_mem = self.query_one("#gpu-mem-input", Input).value.strip()

        # --- Validation ---
        if not name:
            self.notify("Config name is required.", severity="error")
            return
        if not _validate_name(name):
            self.notify(
                "Name must contain only letters, digits, dashes, or underscores.",
                severity="error",
            )
            return

        # --- Build and save ---
        if self._edit_mode:
            cfg = load_config(self._config_name)
            cfg.model = model
            cfg.gpu_memory_utilization = gpu_mem or "0.9"
        else:
            cfg = Config(
                name=name,
                model=model,
                gpu_memory_utilization=gpu_mem or "0.9",
            )

        save_config(cfg)
        self.dismiss(name)

    @on(Button.Pressed, "#cancel-btn")
    def _on_cancel(self, event: Button.Pressed) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------------------
# ConfigListScreen
# ---------------------------------------------------------------------------


class ConfigListScreen(Screen):
    """Full screen listing all configs in a DataTable."""

    BINDINGS = [
        Binding("n", "new_config", "New", show=True),
        Binding("e", "edit_config", "Edit", show=True),
        Binding("delete", "delete_config", "Delete", show=True),
        Binding("escape", "go_back", "Back", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("[b]Configs[/b]", id="config-title")
        yield DataTable(id="config-table", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#config-table", DataTable)
        table.add_columns("Name", "Model", "GPU Mem")
        self._refresh_table()

    def _refresh_table(self) -> None:
        table = self.query_one("#config-table", DataTable)
        table.clear()
        for name in list_config_names():
            cfg = load_config(name)
            model_short = cfg.model.split("/")[-1] if cfg.model else ""
            table.add_row(cfg.name, model_short, cfg.gpu_memory_utilization, key=cfg.name)

    def _get_selected_config(self) -> str | None:
        table = self.query_one("#config-table", DataTable)
        if table.row_count == 0:
            return None
        try:
            row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
            return str(row_key)
        except Exception:
            return None

    # ----- Actions -----

    def action_new_config(self) -> None:
        self.app.push_screen(ConfigFormScreen(), callback=self._on_form_closed)

    def action_edit_config(self) -> None:
        name = self._get_selected_config()
        if name is None:
            self.notify("No config selected.", severity="warning")
            return
        self.app.push_screen(ConfigFormScreen(config_name=name), callback=self._on_form_closed)

    def action_delete_config(self) -> None:
        name = self._get_selected_config()
        if name is None:
            self.notify("No config selected.", severity="warning")
            return
        delete_config(name)
        self.notify(f"Deleted config: {name}")
        self._refresh_table()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def _on_form_closed(self, result: str | None = None) -> None:
        self._refresh_table()
