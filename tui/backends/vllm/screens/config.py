"""Config management screens - form for create/edit and list screen."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.screen import Screen, ModalScreen
from textual.widgets import Button, Static, Label, Input, DataTable, Footer, Header
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.binding import Binding
from textual.suggester import SuggestFromList
from textual import on

from textual import work

from tui.backends.vllm.backend import (
    Config,
    load_config,
    save_config,
    delete_config,
    list_config_names,
    list_profile_names,
    load_profile,
    validate_name as _validate_name,
    extract_vllm_params,
    format_config_param_value,
    parse_config_param_value,
)


# Fallback params used when dynamic extraction fails
_FALLBACK_VLLM_PARAMS: set[str] = {
    "max-model-len", "dtype", "quantization", "load-format",
    "trust-remote-code", "download-dir", "tokenizer", "tokenizer-mode",
    "revision", "code-revision", "tokenizer-revision",
    "served-model-name", "chat-template",
    "max-num-seqs", "max-num-batched-tokens", "max-paddings",
    "scheduling-policy", "preemption-mode",
    "num-scheduler-steps", "multi-step-stream-outputs",
    "swap-space", "kv-cache-dtype", "block-size",
    "enable-prefix-caching", "disable-sliding-window",
    "enforce-eager", "enable-chunked-prefill",
    "disable-async-output-proc", "max-parallel-loading-workers",
    "distributed-executor-backend",
    "max-loras", "max-lora-rank", "lora-extra-vocab-size",
    "long-lora-scaling-factors",
    "speculative-model", "num-speculative-tokens",
    "speculative-max-model-len",
    "disable-log-requests", "disable-log-stats",
    "uvicorn-log-level",
    "seed", "max-logprobs", "response-role",
    "enable-auto-tool-choice", "tool-call-parser",
    "disable-frontend-multiprocessing",
    "otlp-traces-endpoint", "collect-detailed-traces",
    "rope-scaling", "rope-theta",
    "pipeline-parallel-size",
    "reasoning-parser", "mm-encoder-tp-mode",
    "enable-expert-parallel", "mm-processor-cache-type",
}

# Mutable set: starts with fallback, updated dynamically from image
KNOWN_VLLM_PARAMS: set[str] = set(_FALLBACK_VLLM_PARAMS)

_PARAM_SUGGESTER = SuggestFromList(sorted(KNOWN_VLLM_PARAMS), case_sensitive=False)


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
        border: round $primary;
        padding: 1 2;
        width: 90%;
        max-width: 80;
        min-width: 50;
        height: 95%;
        max-height: 40;
        min-height: 12;
    }
    ConfigFormScreen #form-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        width: 100%;
        height: 1;
        margin-bottom: 1;
    }
    ConfigFormScreen .form-row {
        height: auto;
        margin-bottom: 1;
    }
    ConfigFormScreen .form-row Label {
        width: 24;
        padding: 1 1 0 0;
        color: $text-muted;
    }
    ConfigFormScreen #params-title {
        margin-top: 1;
        text-style: bold;
        color: $text;
        border-top: solid $primary 40%;
        padding-top: 1;
    }
    ConfigFormScreen #params-hint {
        color: $text-muted;
        margin-bottom: 1;
    }
    ConfigFormScreen .param-row {
        height: auto;
        margin-bottom: 0;
    }
    ConfigFormScreen .param-row .param-key {
        width: 28;
        margin-right: 1;
    }
    ConfigFormScreen .param-row .param-value {
        width: 1fr;
        margin-right: 1;
    }
    ConfigFormScreen .param-row .param-remove {
        min-width: 5;
        width: 5;
        background: $error 20%;
        border: none;
        color: $error;
    }
    ConfigFormScreen VerticalScroll {
        height: 1fr;
        min-height: 3;
    }
    ConfigFormScreen .form-buttons {
        dock: bottom;
        height: 3;
        padding-top: 1;
        align: center middle;
        background: $surface;
    }
    """

    def __init__(self, config_name: str = "") -> None:
        super().__init__()
        self._config_name = config_name
        self._edit_mode = bool(config_name)
        self._param_counter = 0
        self._initial_config: Config | None = None
        self._saved_name: str | None = None

    def compose(self) -> ComposeResult:
        if self._edit_mode:
            self._initial_config = load_config(self._config_name)
        cfg = self._initial_config

        title = f"Edit Config: {self._config_name}" if self._edit_mode else "New Config"

        with Vertical():
            yield Static(f"[b]{title}[/b]", id="form-title")

            with VerticalScroll():
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

                yield Static("vLLM Parameters", id="params-title")
                yield Static(
                    "[dim]max-model-len, dtype, trust-remote-code, ...[/dim]",
                    id="params-hint",
                )

                yield Vertical(id="params-container")
                yield Button("+ Add Parameter", id="add-param-btn", variant="default")

            with Horizontal(classes="form-buttons"):
                yield Button("Save", id="save-btn", variant="primary")
                yield Button("Close", id="cancel-btn", variant="default")

    def on_mount(self) -> None:
        if self._edit_mode and self._initial_config:
            for key, value in self._initial_config.extra_params.items():
                self._add_param_row(key, format_config_param_value(value))
        self._load_vllm_params()

    @work(exclusive=False)
    async def _load_vllm_params(self) -> None:
        """Load vllm params from docker image and update suggestions."""
        global KNOWN_VLLM_PARAMS, _PARAM_SUGGESTER
        extracted = await extract_vllm_params()
        if extracted:
            KNOWN_VLLM_PARAMS.update(extracted)
            _PARAM_SUGGESTER = SuggestFromList(sorted(KNOWN_VLLM_PARAMS), case_sensitive=False)
            for inp in self.query(".param-key"):
                inp.suggester = _PARAM_SUGGESTER

    def _add_param_row(self, key: str = "", value: str = "") -> None:
        container = self.query_one("#params-container")
        row_id = f"param-row-{self._param_counter}"
        self._param_counter += 1
        row = Horizontal(
            Input(
                value=key,
                placeholder="param-name (Tab: autocomplete)",
                suggester=_PARAM_SUGGESTER,
                classes="param-key",
            ),
            Input(value=value, placeholder="value", classes="param-value"),
            Button("x", classes="param-remove"),
            id=row_id,
            classes="param-row",
        )
        container.mount(row)
        # Scroll to show newly added row
        self.call_after_refresh(self._scroll_to_bottom)

    def _scroll_to_bottom(self) -> None:
        try:
            scroll = self.query_one(VerticalScroll)
            scroll.scroll_end(animate=False)
        except Exception:
            pass

    @on(Button.Pressed, "#add-param-btn")
    def _on_add_param(self, event: Button.Pressed) -> None:
        self._add_param_row()

    @on(Button.Pressed, ".param-remove")
    def _on_remove_param(self, event: Button.Pressed) -> None:
        widget = event.button.parent
        while widget is not None:
            if hasattr(widget, "classes") and "param-row" in widget.classes:
                widget.remove()
                return
            widget = widget.parent

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
        if not self._edit_mode and name in list_config_names():
            self.notify(f"Config '{name}' already exists.", severity="error")
            return
        if gpu_mem:
            try:
                gpu_mem_val = float(gpu_mem)
                if not (0.0 < gpu_mem_val <= 1.0):
                    raise ValueError
            except ValueError:
                self.notify(
                    "GPU Memory Utilization must be between 0.0 and 1.0",
                    severity="error",
                )
                return

        # --- Collect extra params ---
        extra_params: dict[str, Any] = {}
        seen_keys: set[str] = set()
        for row in self.query(".param-row"):
            key_input = row.query_one(".param-key", Input)
            value_input = row.query_one(".param-value", Input)
            k = key_input.value.strip()
            v = value_input.value.strip()
            if k:
                if k in seen_keys:
                    self.notify(f"Duplicate parameter: {k}", severity="error")
                    return
                seen_keys.add(k)
                try:
                    extra_params[k] = parse_config_param_value(v)
                except Exception as exc:
                    self.notify(f"Invalid value for {k}: {exc}", severity="error")
                    return

        # --- Warn about unknown params ---
        unknown = [k for k in extra_params if k not in KNOWN_VLLM_PARAMS]
        if unknown:
            self.notify(
                f"Unknown params (may be valid for your vLLM version): {', '.join(unknown)}",
                severity="warning",
                timeout=6,
            )

        # --- Build and save ---
        cfg = Config(
            name=name,
            model=model,
            gpu_memory_utilization=gpu_mem or "0.9",
            extra_params=extra_params,
        )

        save_config(cfg)
        self.notify(f"Saved: {name}", severity="information")
        self._saved_name = name

        # New config → switch to edit mode after first save
        if not self._edit_mode:
            self._edit_mode = True
            self._config_name = name
            self.query_one("#name-input", Input).disabled = True
            self.query_one("#form-title", Static).update(f"[b]Edit Config: {name}[/b]")

    @on(Button.Pressed, "#cancel-btn")
    def _on_close(self, event: Button.Pressed) -> None:
        self.dismiss(self._saved_name)

    def action_cancel(self) -> None:
        self.dismiss(self._saved_name)


# ---------------------------------------------------------------------------
# ConfirmDeleteConfigScreen
# ---------------------------------------------------------------------------


class ConfirmDeleteConfigScreen(ModalScreen[bool]):
    """Confirmation modal before deleting a config."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    DEFAULT_CSS = """
    ConfirmDeleteConfigScreen {
        align: center middle;
    }
    ConfirmDeleteConfigScreen > Vertical {
        background: $surface;
        border: round $error;
        padding: 1 2;
        width: 50;
        height: auto;
    }
    ConfirmDeleteConfigScreen #confirm-title {
        text-style: bold;
        color: $error;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }
    ConfirmDeleteConfigScreen #confirm-msg {
        text-align: center;
        margin-bottom: 1;
    }
    ConfirmDeleteConfigScreen #confirm-warn {
        color: $warning;
        text-align: center;
        margin-bottom: 1;
    }
    ConfirmDeleteConfigScreen .confirm-buttons {
        height: auto;
        align: center middle;
    }
    ConfirmDeleteConfigScreen .confirm-buttons Button {
        margin: 0 1;
    }
    """

    def __init__(self, config_name: str, referencing_profiles: list[str]) -> None:
        super().__init__()
        self._config_name = config_name
        self._referencing = referencing_profiles

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Delete Config", id="confirm-title")
            yield Static(
                f"Are you sure you want to delete [b]{self._config_name}[/b]?",
                id="confirm-msg",
            )
            if self._referencing:
                names = ", ".join(self._referencing)
                yield Static(
                    f"[b]Warning:[/b] Used by profiles: {names}",
                    id="confirm-warn",
                )
            with Horizontal(classes="confirm-buttons"):
                yield Button("Delete", id="confirm-yes", variant="error")
                yield Button("Cancel", id="confirm-no", variant="default")

    @on(Button.Pressed, "#confirm-yes")
    def _on_yes(self, event: Button.Pressed) -> None:
        from tui.backends.vllm.backend import save_profile
        for profile_name in self._referencing:
            p = load_profile(profile_name)
            p.config_name = ""
            save_profile(p)
        delete_config(self._config_name)
        self.app.notify(f"Deleted config: {self._config_name}")
        self.dismiss(True)

    @on(Button.Pressed, "#confirm-no")
    def _on_no(self, event: Button.Pressed) -> None:
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)


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
        table.add_columns("Name", "Model", "GPU Mem", "Params")
        self._refresh_table()

    def _refresh_table(self) -> None:
        table = self.query_one("#config-table", DataTable)
        table.clear()
        for name in list_config_names():
            cfg = load_config(name)
            model_short = cfg.model.split("/")[-1] if cfg.model else ""
            param_count = str(len(cfg.extra_params)) if cfg.extra_params else ""
            table.add_row(cfg.name, model_short, cfg.gpu_memory_utilization, param_count, key=cfg.name)

    def _get_selected_config(self) -> str | None:
        table = self.query_one("#config-table", DataTable)
        if table.row_count == 0:
            return None
        try:
            row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
            return row_key.value
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
        # Check if any profile references this config
        referencing = [
            p for p in list_profile_names()
            if load_profile(p).config_name == name
        ]
        self.app.push_screen(
            ConfirmDeleteConfigScreen(name, referencing),
            callback=self._on_delete_confirmed,
        )

    def _on_delete_confirmed(self, result: bool) -> None:
        if result:
            self._refresh_table()

    def action_go_back(self) -> None:
        self.app.switch_screen("dashboard")

    def _on_form_closed(self, result: str | None = None) -> None:
        self._refresh_table()
