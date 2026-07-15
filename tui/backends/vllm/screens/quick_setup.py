"""Quick Setup screen - create profile + config in one step."""

from __future__ import annotations

import re
from typing import Any

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static, Switch, Select

from tui.common import profile_store
from tui.common.i18n import t


class QuickSetupScreen(ModalScreen[str]):
    """Quick setup: create a profile + config from a model name."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("pageup", "scroll_form('up')", "Scroll up", show=False),
        Binding("pagedown", "scroll_form('down')", "Scroll down", show=False),
        Binding("home", "scroll_form('home')", "Scroll to top", show=False),
        Binding("end", "scroll_form('end')", "Scroll to bottom", show=False),
    ]

    DEFAULT_CSS = """
    QuickSetupScreen {
        align: center middle;
    }
    QuickSetupScreen > Vertical {
        background: $surface;
        border: round $primary;
        padding: 1 2;
        width: 90%;
        max-width: 70;
        min-width: 45;
        /* No max-height — see ProfileFormScreen for rationale. */
        height: 95%;
        min-height: 12;
    }
    QuickSetupScreen VerticalScroll {
        height: 1fr;
        min-height: 3;
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
        height: auto;
        min-height: 3;
        margin-top: 1;
        padding-top: 1;
        align: center middle;
        background: $surface;
        border-top: solid $primary 30%;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._last_estimated_model = ""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(t("Quick Setup", "빠른 설정"), classes="title")
            with VerticalScroll():
                yield Label(t("HuggingFace Model (e.g., meta-llama/Llama-3-8B)", "HuggingFace 모델 (예: meta-llama/Llama-3-8B)"))
                yield Input(placeholder="org/model-name", id="model-input")
                yield Static("", id="mem-estimate")
                yield Label("GPU ID")
                yield Input(placeholder="0", value="0", id="gpu-input")
                yield Label("Port")
                # Prefill from the effective default (profiles.yaml `defaults:`
                # can override the built-in 8000) so the form agrees with what
                # `profile new --port 0` / quick-setup would pick.
                _port = str(profile_store.effective_defaults("vllm")["port"])
                yield Input(placeholder=_port, value=_port, id="port-input")
                yield Label(t("GPU Memory Utilization", "GPU 메모리 사용률"))
                yield Input(placeholder="0.9", value="0.9", id="gpu-mem-input")
                yield Label(t("Copy params from (optional)", "파라미터 복사 원본 (선택)"))
                yield Select(
                    self._build_config_options(),
                    prompt=t("None", "없음"),
                    allow_blank=True,
                    id="copy-config-select",
                )
                with Horizontal():
                    yield Label(t("Enable LoRA", "LoRA 사용"))
                    yield Switch(id="lora-switch")
            with Horizontal(classes="buttons"):
                yield Button(t("Create", "생성"), variant="primary", id="create-btn")
                yield Button(t("Cancel", "취소"), id="cancel-btn")

    def _build_config_options(self) -> list[tuple[str, str]]:
        from tui.backends.vllm.backend import list_config_names, load_config

        return [
            (t(f"{name} ({len(load_config(name).extra_params)} params)", f"{name} ({len(load_config(name).extra_params)} 파라미터)"), name)
            for name in list_config_names()
        ]

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
            label.update(t(f"[dim]Estimating memory for {model_id}...[/dim]", f"[dim]{model_id} 메모리 추정 중...[/dim]"))
        except Exception:
            return
        from tui.backends.vllm.backend import estimate_model_memory
        result = await estimate_model_memory(model_id)
        try:
            self.query_one("#mem-estimate", Static).update(
                t(f"[bold]Est. Memory:[/bold] {result}", f"[bold]예상 메모리:[/bold] {result}")
            )
        except Exception:
            pass

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel(self) -> None:
        self.dismiss("")

    def action_cancel(self) -> None:
        self.dismiss("")

    def action_scroll_form(self, direction: str) -> None:
        try:
            scroll = self.query_one(VerticalScroll)
        except Exception:
            return
        if direction == "up":
            scroll.scroll_page_up()
        elif direction == "down":
            scroll.scroll_page_down()
        elif direction == "home":
            scroll.scroll_home()
        elif direction == "end":
            scroll.scroll_end()

    @on(Button.Pressed, "#create-btn")
    def on_create(self) -> None:
        model = self.query_one("#model-input", Input).value.strip()
        gpu = self.query_one("#gpu-input", Input).value.strip()
        port = self.query_one("#port-input", Input).value.strip()
        gpu_mem = self.query_one("#gpu-mem-input", Input).value.strip()
        lora = self.query_one("#lora-switch", Switch).value

        if not model:
            self.notify(t("Model name is required", "모델 이름은 필수입니다"), severity="error")
            return

        # Derive name from model
        name_part = model.rsplit("/", 1)[-1]
        safe_name = re.sub(r"[^a-zA-Z0-9-]", "-", name_part).lower().strip("-")

        if not safe_name:
            self.notify(t("Could not derive a valid name from model", "모델에서 유효한 이름을 만들 수 없습니다"),
                        severity="error")
            return

        # Validate port
        try:
            port_num = int(port)
            if not 1024 <= port_num <= 65535:
                raise ValueError
        except ValueError:
            self.notify(t("Port must be between 1024 and 65535", "Port 는 1024 ~ 65535 사이여야 합니다"),
                        severity="error")
            return

        # Validate GPU Memory Utilization
        if gpu_mem:
            try:
                gpu_mem_val = float(gpu_mem)
                if not (0.0 < gpu_mem_val <= 1.0):
                    raise ValueError
            except ValueError:
                self.notify(t("GPU Memory Utilization must be between 0.0 and 1.0", "GPU 메모리 사용률은 0.0 ~ 1.0 사이여야 합니다"),
                            severity="error")
                return

        # Validate GPU
        if not gpu or not re.match(r"^[0-9]+(,[0-9]+)*$", gpu):
            self.notify(t("GPU ID is required (e.g., 0 or 0,1)", "GPU ID 는 필수입니다 (예: 0 또는 0,1)"),
                        severity="error")
            return

        # Calculate tensor parallel from GPU count
        gpu_count = len(gpu.split(","))

        from tui.backends.vllm.backend import (
            Profile, Config, save_profile, save_config,
            list_profile_names, list_config_names, load_config,
        )

        # Auto-resolve name collision by appending suffix. `example` is added
        # explicitly because list_config_names() filters it out — without it a
        # model named "example" would overwrite the tracked example.yaml.
        existing_profiles = list_profile_names()
        existing_configs = set(list_config_names()) | {"example"}
        original_name = safe_name
        suffix = 0
        while safe_name in existing_profiles or safe_name in existing_configs:
            suffix += 1
            safe_name = f"{original_name}-{suffix}"

        # Copy extra params from selected config
        extra_params: dict[str, Any] = {}
        copy_select = self.query_one("#copy-config-select", Select)
        if copy_select.value and copy_select.value != Select.BLANK:
            source_cfg = load_config(str(copy_select.value))
            extra_params = dict(source_cfg.extra_params)

        # Save config
        config = Config(
            name=safe_name,
            model=model,
            gpu_memory_utilization=gpu_mem or "0.9",
            extra_params=extra_params,
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
            model_id=model,
            enable_lora="true" if lora else "false",
        )
        save_profile(profile)

        self.notify(
            t(
                f"✓ Created: {safe_name}  (next: press 'u' to start)",
                f"✓ 생성: {safe_name}  (다음: 'u' 로 시작)",
            ),
            severity="information",
        )
        self.dismiss(safe_name)
