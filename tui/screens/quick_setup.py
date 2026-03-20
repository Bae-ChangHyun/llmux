"""Quick Setup screen - create profile + config in one step."""

from __future__ import annotations

import re

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static, Switch


class QuickSetupScreen(ModalScreen[str]):
    """Quick setup: create a profile + config from a model name."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    DEFAULT_CSS = """
    QuickSetupScreen {
        align: center middle;
    }
    QuickSetupScreen > Vertical {
        background: $surface;
        border: round $primary;
        padding: 1 2;
        width: 65;
        max-height: 90%;
    }
    QuickSetupScreen .title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
        text-align: center;
        width: 100%;
    }
    QuickSetupScreen Label {
        margin-top: 1;
        color: $text-muted;
    }
    QuickSetupScreen Input {
        margin-bottom: 0;
    }
    QuickSetupScreen #mem-estimate {
        height: 1;
        color: $text-muted;
        margin-top: 0;
    }
    QuickSetupScreen .buttons {
        height: 1;
        align: center middle;
        margin-top: 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._last_estimated_model = ""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Quick Setup", classes="title")
            with VerticalScroll():
                yield Label("HuggingFace Model (e.g., meta-llama/Llama-3-8B)")
                yield Input(placeholder="org/model-name", id="model-input")
                yield Static("", id="mem-estimate")
                yield Label("GPU ID")
                yield Input(placeholder="0", value="0", id="gpu-input")
                yield Label("Port")
                yield Input(placeholder="8000", value="8000", id="port-input")
                yield Label("GPU Memory Utilization")
                yield Input(placeholder="0.9", value="0.9", id="gpu-mem-input")
                with Horizontal():
                    yield Label("Enable LoRA")
                    yield Switch(id="lora-switch")
            with Horizontal(classes="buttons"):
                yield Button("Create", variant="primary", id="create-btn")
                yield Button("Cancel", id="cancel-btn")

    @on(Input.Blurred, "#model-input")
    def _on_model_blur(self, event: Input.Blurred) -> None:
        model = event.input.value.strip()
        if model and model != self._last_estimated_model:
            self._last_estimated_model = model
            self._estimate_memory(model)

    @work(exclusive=True, group="mem-estimate")
    async def _estimate_memory(self, model_id: str) -> None:
        try:
            label = self.query_one("#mem-estimate", Static)
            label.update(f"[dim]Estimating memory for {model_id}...[/dim]")
        except Exception:
            return
        from tui.backend import estimate_model_memory
        result = await estimate_model_memory(model_id)
        try:
            self.query_one("#mem-estimate", Static).update(
                f"[bold]Est. Memory:[/bold] {result}"
            )
        except Exception:
            pass

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel(self) -> None:
        self.dismiss("")

    def action_cancel(self) -> None:
        self.dismiss("")

    @on(Button.Pressed, "#create-btn")
    def on_create(self) -> None:
        model = self.query_one("#model-input", Input).value.strip()
        gpu = self.query_one("#gpu-input", Input).value.strip()
        port = self.query_one("#port-input", Input).value.strip()
        gpu_mem = self.query_one("#gpu-mem-input", Input).value.strip()
        lora = self.query_one("#lora-switch", Switch).value

        if not model:
            self.notify("Model name is required", severity="error")
            return

        # Derive name from model
        name_part = model.rsplit("/", 1)[-1]
        safe_name = re.sub(r"[^a-zA-Z0-9-]", "-", name_part).lower().strip("-")

        if not safe_name:
            self.notify("Could not derive a valid name from model", severity="error")
            return

        # Validate port
        try:
            port_num = int(port)
            if not 1024 <= port_num <= 65535:
                raise ValueError
        except ValueError:
            self.notify("Port must be between 1024 and 65535", severity="error")
            return

        # Validate GPU Memory Utilization
        if gpu_mem:
            try:
                gpu_mem_val = float(gpu_mem)
                if not (0.0 < gpu_mem_val <= 1.0):
                    raise ValueError
            except ValueError:
                self.notify("GPU Memory Utilization must be between 0.0 and 1.0", severity="error")
                return

        # Validate GPU
        if not gpu or not re.match(r"^[0-9]+(,[0-9]+)*$", gpu):
            self.notify("GPU ID is required (e.g., 0 or 0,1)", severity="error")
            return

        # Calculate tensor parallel from GPU count
        gpu_count = len(gpu.split(","))

        from tui.backend import (
            Profile, Config, save_profile, save_config,
            list_profile_names, list_config_names,
        )

        # Auto-resolve name collision by appending suffix
        existing_profiles = list_profile_names()
        existing_configs = list_config_names()
        original_name = safe_name
        suffix = 1
        while safe_name in existing_profiles or safe_name in existing_configs:
            suffix += 1
            safe_name = f"{original_name}-{suffix}"

        # Save config
        config = Config(
            name=safe_name,
            model=model,
            gpu_memory_utilization=gpu_mem or "0.9",
        )
        save_config(config)

        # Save profile
        profile = Profile(
            name=safe_name,
            container_name=safe_name,
            port=port,
            gpu_id=gpu,
            tensor_parallel=str(gpu_count),
            config_name=safe_name,
            enable_lora="true" if lora else "false",
        )
        save_profile(profile)

        self.notify(f"Created profile + config: {safe_name}", severity="information")
        self.dismiss(safe_name)
