"""Profile management screens - form for create/edit and delete confirmation."""

from __future__ import annotations

import re

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Button, Static, Label, Input, Select, Switch
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual import on

from tui.backend import (
    Profile,
    load_profile,
    save_profile,
    delete_profile,
    list_config_names,
    list_profile_names,
    validate_name as _validate_name,
)


# ---------------------------------------------------------------------------
# ProfileFormScreen
# ---------------------------------------------------------------------------


class ProfileFormScreen(ModalScreen[str | None]):
    """Modal form for creating or editing a profile.

    Pass a Profile object to edit it; omit for a new blank form.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    DEFAULT_CSS = """
    ProfileFormScreen {
        align: center middle;
    }
    ProfileFormScreen > Vertical {
        background: $surface;
        border: round $primary;
        padding: 1 2;
        width: 60;
        max-height: 85%;
    }
    ProfileFormScreen #form-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }
    ProfileFormScreen .form-row {
        height: auto;
        margin-bottom: 1;
    }
    ProfileFormScreen .form-row Label {
        width: 22;
        padding: 1 1 0 0;
        color: $text-muted;
    }
    ProfileFormScreen #env-vars-section {
        margin-top: 1;
        border-top: solid $primary 40%;
        padding-top: 1;
    }
    ProfileFormScreen #env-vars-title {
        text-style: bold;
        color: $text;
    }
    ProfileFormScreen .env-var-line {
        color: $text-muted;
        height: auto;
        margin-left: 2;
    }
    ProfileFormScreen .form-buttons {
        height: 1;
        margin-top: 1;
        align: center middle;
    }
    """

    def __init__(self, profile: Profile | None = None) -> None:
        super().__init__()
        self._profile = profile
        self._edit_mode = profile is not None
        self._saved_name: str | None = None

    def compose(self) -> ComposeResult:
        p = self._profile
        title = f"Edit Profile: {p.name}" if self._edit_mode else "New Profile"

        configs = list_config_names()
        config_options: list[tuple[str, str]] = [(name, name) for name in configs]

        with Vertical():
            yield Static(f"[b]{title}[/b]", id="form-title")

            with VerticalScroll():
                with Horizontal(classes="form-row"):
                    yield Label("Profile Name")
                    yield Input(
                        value=p.name if p else "",
                        placeholder="my-profile",
                        id="name-input",
                        disabled=self._edit_mode,
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Container Name")
                    yield Input(
                        value=p.container_name if p else "",
                        placeholder="container-name",
                        id="container-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Port")
                    yield Input(
                        value=p.port if p else "",
                        placeholder="8000",
                        id="port-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("GPU ID")
                    yield Input(
                        value=p.gpu_id if p else "",
                        placeholder="0",
                        id="gpu-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Tensor Parallel")
                    yield Input(
                        value=p.tensor_parallel if p else "",
                        placeholder="1",
                        id="tp-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Config")
                    select_kwargs: dict = dict(
                        prompt="Select config",
                        allow_blank=True,
                        id="config-select",
                    )
                    if p and p.config_name and p.config_name in configs:
                        select_kwargs["value"] = p.config_name
                    yield Select(config_options, **select_kwargs)

                with Horizontal(classes="form-row"):
                    yield Label("Model ID")
                    yield Input(
                        value=(p.model_id if p else ""),
                        placeholder="org/model-name (used for auto config)",
                        id="model-id-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Enable LoRA")
                    yield Switch(
                        value=(p.enable_lora == "true") if p else False,
                        id="lora-switch",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Extra Pip Packages")
                    extra_pkgs = (p.env_vars.get("EXTRA_PIP_PACKAGES", "") if p else "")
                    yield Input(
                        value=extra_pkgs,
                        placeholder="e.g. flash-attn bitsandbytes",
                        id="extra-pip-input",
                    )

            with Horizontal(classes="form-buttons"):
                yield Button("Save", id="save-btn", variant="primary")
                yield Button("Close", id="cancel-btn", variant="default")

    @on(Button.Pressed, "#save-btn")
    def _on_save(self, event: Button.Pressed) -> None:
        name = self.query_one("#name-input", Input).value.strip()
        container = self.query_one("#container-input", Input).value.strip()
        port = self.query_one("#port-input", Input).value.strip()
        gpu_id = self.query_one("#gpu-input", Input).value.strip()
        tp = self.query_one("#tp-input", Input).value.strip()
        lora = self.query_one("#lora-switch", Switch).value

        config_select = self.query_one("#config-select", Select)
        config_name = str(config_select.value) if config_select.value != Select.BLANK else ""
        model_id = self.query_one("#model-id-input", Input).value.strip()

        # --- Validation ---
        if not name:
            self.notify("Profile name is required.", severity="error")
            return
        if not _validate_name(name):
            self.notify(
                "Name must start with a letter/digit, and contain only letters, digits, dashes, or underscores.",
                severity="error",
            )
            return

        if not self._edit_mode and name in list_profile_names():
            self.notify(f"Profile '{name}' already exists.", severity="error")
            return

        if container and not _validate_name(container):
            self.notify(
                "Container name must contain only letters, digits, dashes, or underscores.",
                severity="error",
            )
            return

        if port:
            try:
                port_int = int(port)
                if not (1024 <= port_int <= 65535):
                    raise ValueError
            except ValueError:
                self.notify("Port must be a number between 1024 and 65535.", severity="error")
                return

        if gpu_id and not re.match(r"^[\d,]+$", gpu_id):
            self.notify("GPU ID must contain only digits and commas.", severity="error")
            return

        if tp:
            try:
                tp_int = int(tp)
                if tp_int < 1:
                    raise ValueError
            except ValueError:
                self.notify("Tensor Parallel must be a positive integer.", severity="error")
                return

        extra_pip = self.query_one("#extra-pip-input", Input).value.strip()

        # --- Build and save ---
        if self._edit_mode and self._profile is not None:
            profile = self._profile
            profile.container_name = container or name
            profile.port = port or "8000"
            profile.gpu_id = gpu_id or "0"
            profile.tensor_parallel = tp or "1"
            profile.config_name = config_name
            profile.model_id = model_id
            profile.enable_lora = "true" if lora else "false"
        else:
            profile = Profile(
                name=name,
                container_name=container or name,
                port=port or "8000",
                gpu_id=gpu_id or "0",
                tensor_parallel=tp or "1",
                config_name=config_name,
                model_id=model_id,
                enable_lora="true" if lora else "false",
            )

        # Handle EXTRA_PIP_PACKAGES
        if extra_pip:
            profile.env_vars["EXTRA_PIP_PACKAGES"] = extra_pip
        else:
            profile.env_vars.pop("EXTRA_PIP_PACKAGES", None)

        save_profile(profile)
        self.notify(f"Saved: {name}", severity="information")
        self._saved_name = name

        # New profile → switch to edit mode after first save
        if not self._edit_mode:
            self._edit_mode = True
            self._profile = profile
            self.query_one("#name-input", Input).disabled = True
            self.query_one("#form-title", Static).update(f"[b]Edit Profile: {name}[/b]")

    @on(Button.Pressed, "#cancel-btn")
    def _on_close(self, event: Button.Pressed) -> None:
        self.dismiss(self._saved_name)

    def action_cancel(self) -> None:
        self.dismiss(self._saved_name)


# ---------------------------------------------------------------------------
# ProfileDeleteScreen
# ---------------------------------------------------------------------------


class ProfileDeleteScreen(ModalScreen[bool]):
    """Simple confirmation modal for deleting a profile."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    DEFAULT_CSS = """
    ProfileDeleteScreen {
        align: center middle;
    }
    ProfileDeleteScreen > Vertical {
        background: $surface;
        border: round $error;
        padding: 1 2;
        width: 50;
        height: auto;
    }
    ProfileDeleteScreen #delete-message {
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }
    ProfileDeleteScreen .form-buttons {
        height: 1;
        margin-top: 1;
        align: center middle;
    }
    """

    def __init__(self, profile_name: str) -> None:
        super().__init__()
        self._profile_name = profile_name
        self._profile = load_profile(profile_name)

    def compose(self) -> ComposeResult:
        with Vertical():
            if self._profile.config_name:
                yield Static(
                    f"Delete [b]{self._profile_name}[/b]?\n"
                    f"(profile + config: {self._profile.config_name})",
                    id="delete-message",
                )
            else:
                yield Static(
                    f"Delete profile [b]{self._profile_name}[/b]?",
                    id="delete-message",
                )
            with Horizontal(classes="form-buttons"):
                yield Button("Delete", id="delete-btn", variant="error")
                yield Button("Cancel", id="cancel-btn", variant="default")

    @on(Button.Pressed, "#delete-btn")
    def _on_delete(self, event: Button.Pressed) -> None:
        has_config = bool(self._profile.config_name)
        delete_profile(self._profile_name, delete_config=has_config)
        if has_config:
            self.app.notify(f"Deleted: {self._profile_name} + config: {self._profile.config_name}")
        else:
            self.app.notify(f"Deleted profile: {self._profile_name}")
        self.dismiss(True)

    @on(Button.Pressed, "#cancel-btn")
    def _on_cancel(self, event: Button.Pressed) -> None:
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)
